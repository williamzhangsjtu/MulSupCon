import torch
import torch.nn as nn
import h5py
import fire
import os
import numpy as np
import datetime
from glob import glob
from tqdm import tqdm

import utils
import losses
import schedulers
from metrics import get_evaluate_metrics,\
    get_mlc_metrics
from dataloader import create_dataloader

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint,\
    EarlyStopping, global_step_from_engine
from ignite.metrics import RunningAverage, Loss

class Runner(object):

    def __init__(self, seed=0):
        super(Runner, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        self.seed = seed
    
    
    @staticmethod
    def _forward(model, batch, device=0):
        inputs, targets = batch
        inputs, targets = inputs.cuda(device), targets.cuda(device)
        return model(inputs, targets) # score, mask

    def pretrain(
        self,
        config: str = 'config/pretrain_yeast.yaml',
        debug: bool = False,
        resume: bool = False,
        model_path: str = None,
        is_image: bool = False,
        load_level: str = 'all',
        **kwargs):
        """
        pretrain on Multi-label Dataset

        Parameters:
            config (str): path to the configuration file
            debug (bool): debug mode
            resume (bool): resume from the saved model
            model_path (str): path to the saved model
            is_image (bool): whether the input is image
            load_level (str): which part of the model to load
        """
        assert load_level in ['all', 'encoder', 'backbone']
        if resume:
            assert model_path is not None
            load_level = 'all'
            config = torch.load(
                glob(os.path.join(model_path, '*config*'))[0], map_location='cpu')
            n_class = config['model_args']['n_class']
            for k, v in kwargs.items():
                config[k] = v
            config['scheduler_args']['warmup'] = False
        else:
            config = utils.parse_config(config, debug, **kwargs)
            with h5py.File(config['train'], 'r') as train_input:
                n_class = train_input['target'].shape[-1]
                config['model_args']['n_class'] = n_class
        
        outputdir = os.path.join(config['outputdir'], 
            "{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')))
        os.makedirs(outputdir, exist_ok=False)
        torch.save(config, os.path.join(outputdir, 'run_config.d'))
        logger = utils.Logger(os.path.join(outputdir, 'logging.txt'))
        logger.info('<============== Device Info ==============>')
        logger.info(f'Using device: {torch.cuda.get_device_name(0)}')
        logger.info('<============== Meta Data ==============>')
        logger.info(f'Output directory is: {outputdir}')
        logger.info(f'Using {self.seed} as seed')
        if resume:
            logger.info(f'Resume from {model_path}')
        if model_path:
            logger.info(f'Loading params from {model_path}')
        logger.info('<============== Configuration ==============>')
        for k, v in config.items():
            logger.info(f'{k}: {v}')
        logger.info('<============== Training ==============>')
        
        transform_kwargs = config.get('transform_kwargs', {})
        transform, _ = utils.get_transform(
            is_image=is_image,
            **transform_kwargs
        )

        indices = utils.parse_data(
            config['train'],
            debug=debug,
            seed=self.seed
        )
        TrainDataloader = create_dataloader(
            input_h5=config['train'],
            indices=indices,
            is_eval=False,
            seed=self.seed,
            transform=transform,
            is_image=is_image,
            **config['dataloader_args'])
        
        output_func = utils.get_output_func(**config['pattern_args'])
        model, optim_params, scheduler_params = utils.get_model_from_pretrain(
            model_path=model_path,
            config=config,
            resume=resume,
            output_func=output_func,
            load_level=load_level
        )
        model.cuda(0)
        criterion = getattr(losses, config['criterion'])(**config['criterion_args'])


        if not (resume and optim_params):
            lr = config['optimizer_args'].get('lr', 0.004)
            lrp = config.get('lrp', 1)
            logger.info(f'Using different learning rate: {lr} and {lr*lrp}')
            optimizer = getattr(torch.optim, config['optimizer'])(
                model.get_config_optim(lr, lrp), **config['optimizer_args'])
        else:
            optimizer = getattr(torch.optim, config['optimizer'])(
                model.parameters(), **config['optimizer_args'])

        scheduler = getattr(schedulers, config['scheduler'])(
            optimizer, **config['scheduler_args'])
        if resume and optim_params:
            optimizer.load_state_dict(optim_params)
        if scheduler_params and resume:
            scheduler.load_state_dict(scheduler_params)

        def _train(_, batch):
            model.train()
            with torch.enable_grad():
                score, mask = Runner._forward(model, batch)
                loss = criterion(score, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return loss.cpu().item()

        
        trainer = Engine(_train)
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'Loss')
        ProgressBar(persist=False, ncols=75).attach(
            trainer, output_transform=lambda x: {'loss': x})


        @trainer.on(Events.EPOCH_COMPLETED)
        def evaluate(engine):
            lr = optimizer.param_groups[0]['lr']
            lr = round(lr, 7)
            logger.info(f'<==== Epoch {trainer.state.epoch}, lr {lr} ====>')
            train_loss = engine.state.metrics['Loss']
            logger.info('Training Loss: {:<5.4f}'.format(train_loss))
            loss = train_loss
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(loss)
            else:
                scheduler.step()

        BestModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_best',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=1,
            global_step_transform=global_step_from_engine(trainer))
        PeriodModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_period',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=None,
            global_step_transform=global_step_from_engine(trainer))
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=config['saving_interval']), 
            PeriodModelCheckpoint, 
            {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler,
            })
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            BestModelCheckpoint,
            {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler,
            })

        EarlyStoppingHandler = EarlyStopping(
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=trainer, patience=config['patience'])
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            EarlyStoppingHandler)
    
        trainer.run(TrainDataloader,
            max_epochs=config['n_epochs'], epoch_length=config['iters_per_epoch'])

        return outputdir
            
    def train(
        self, 
        model_path: str = None, 
        resume: bool = False,
        config: str = 'config/train_yeast.yaml', 
        linear_probe: bool = False, 
        debug: bool = False, 
        is_image: bool = False,
        load_level: str = 'backbone',
        **kwargs
    ):
        """
        Tune the pretrained model

        Paremeters:
            model_path (str): path to the saved model
            resume (bool): resume from the saved model
            config (str): path to the configuration file
            linear_probe (bool): whether to use linear probe
            debug (bool): debug mode
            is_image (bool): whether the input is image
            load_level (str): which part of the model to load
        """
        assert load_level in ['all', 'encoder', 'backbone']
        config = utils.parse_config(config, debug, **kwargs)
        with h5py.File(config['train'], 'r') as train_input:
            n_class = train_input['target'].shape[-1]
            config['model_args']['n_class'] = n_class

        config['dev'] = config.get('dev', config['test'])
        train = utils.parse_data(
            config['train'],
            debug=debug,
            seed=self.seed
        )
        dev = utils.parse_data(
            config['dev'],
            debug=debug,
            seed=self.seed
        )
        test = utils.parse_data(
            config['test'],
            debug=debug,
            seed=self.seed
        )
        suffix = 'finetune' if not linear_probe else 'linear'
        if model_path is None:
            outputdir = os.path.join(config['outputdir'], suffix,
                "{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')))
        else:
            outputdir = os.path.join(model_path, suffix,
                "{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')))
        os.makedirs(outputdir, exist_ok=False)
        torch.save(config, os.path.join(outputdir, 'run_config.d'))
        logger = utils.Logger(os.path.join(outputdir, 'logging.txt'))
        logger.info('<============== Device Info ==============>')
        logger.info(f'Using device: {torch.cuda.get_device_name(0)}')
        logger.info('<============== Meta Data ==============>')
        logger.info(f'Output directory is: {outputdir}')
        logger.info(f'Using {self.seed} as seed')
        logger.info(f'Loading params strategy is: {load_level}')
        if model_path:
            logger.info(f'Loading params from {model_path}')
        logger.info('<============== Configuration ==============>')
        for k, v in config.items():
            logger.info(f'{k}: {v}')
        logger.info('<============== Training ==============>')
        print(f"Number of train: {len(train)}")
        print(f"Number of dev: {len(dev)}")
        print(f"Number of test: {len(test)}")

        transform_kwargs = config.get('transform_kwargs', {})
        train_transform, eval_transform = utils.get_transform(
            is_image=is_image,
            **transform_kwargs
        )
        TrainDataloader = create_dataloader(
            input_h5=config['train'],
            indices=train,
            is_eval=False,
            transform=train_transform,
            is_image=is_image,
            seed=self.seed,
            **config['dataloader_args']
        )
        DevDataloader = create_dataloader(
            input_h5=config['dev'],
            indices=dev,
            is_eval=True,
            is_image=is_image,
            transform=eval_transform,
            **config['dataloader_args']
        )
        TestDataloader = create_dataloader(
            input_h5=config['test'],
            indices=test,
            is_eval=True,
            is_image=is_image,
            transform=eval_transform,
            **config['dataloader_args']
        )

        
        model, optim_params, scheduler_params = utils.get_model_from_pretrain(
            model_path=model_path,
            config=config,
            resume=resume,
            load_level=load_level
        )
        model.set_linear_probe(linear_probe)
        model.cuda(0)

        criterion = getattr(losses, config['criterion'])(**config['criterion_args'])

        if linear_probe:
            optimizer = getattr(torch.optim, config['optimizer'])(
                model.parameters(), **config['optimizer_args'])
        else:
            lr = config['optimizer_args'].get('lr', 0.004)
            lrp = config.get('lrp', 1)
            parameters = model.get_config_optim(lr, lrp)
            optimizer = getattr(torch.optim, config['optimizer'])(parameters, **config['optimizer_args'])

        scheduler = getattr(schedulers, config['scheduler'])(
            optimizer, **config['scheduler_args'])
        if resume and optim_params:
            optimizer.load_state_dict(optim_params)
        if resume and scheduler_params:
            scheduler.load_state_dict(scheduler_params)


        def _train(_, batch):
            model.train()
            with torch.enable_grad():
                preds, labels = Runner._forward(model, batch)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return loss.cpu().item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                preds, labels = Runner._forward(model, batch)
            return preds, labels

        
        trainer, evaluator = Engine(_train), Engine(_inference)
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'Loss')
        Loss(criterion).attach(evaluator, 'Loss')
        ProgressBar(persist=False, ncols=75).attach(
            trainer, output_transform=lambda x: {'loss': x})
        ProgressBar(persist=False, ncols=75, desc='Evaluating').attach(
            evaluator, output_transform=None)

        best_dev_loss = [np.inf]
        @trainer.on(Events.EPOCH_COMPLETED)
        def evaluate(engine):
            lr = optimizer.param_groups[0]['lr']
            lr = round(lr, 7)
            logger.info(f'<==== Epoch {trainer.state.epoch}, lr {lr} ====>')
            evaluator.run(DevDataloader)
            train_loss = engine.state.metrics['Loss']
            eval_loss = evaluator.state.metrics['Loss']
            logger.info('Training Loss: {:<5.4f}'.format(train_loss))
            logger.info('Evaluation Loss: {:<5.4f}'.format(eval_loss))

            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(eval_loss)
            else:
                scheduler.step()

            if best_dev_loss[0] > eval_loss:
                best_dev_loss[0] = eval_loss
                test()


        def test():
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in TestDataloader:
                    preds, labels = Runner._forward(model, batch)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
            all_preds, all_labels = torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
            all_labels = all_labels.to(torch.int32)
            
            all_preds = torch.sigmoid(all_preds)
            metrics = get_evaluate_metrics()(all_preds, all_labels)

            all_preds = all_preds.numpy()
            all_labels = all_labels.numpy()
            metrics = {**get_mlc_metrics(all_preds, all_labels), **metrics}
            for name, metric in metrics.items():
                logger.info('{}: {:<5.3f}'.format(name, metric))
            return metrics


        BestModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='eval_best',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=1,
            global_step_transform=global_step_from_engine(trainer))
        PeriodModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_period',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=None,
            global_step_transform=global_step_from_engine(trainer))
        EarlyStoppingHandler = EarlyStopping(
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=trainer, patience=config['patience'])
        

        BestModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='eval_best',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=1,
            global_step_transform=global_step_from_engine(trainer))
        PeriodModelCheckpoint = ModelCheckpoint(
            outputdir, filename_prefix='train_period',
            score_function=lambda engine: -engine.state.metrics['Loss'],
            score_name='Loss', n_saved=None,
            global_step_transform=global_step_from_engine(trainer))
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=config['saving_interval']), 
            PeriodModelCheckpoint, 
            {'model': model, 'optimizer': optimizer})
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            BestModelCheckpoint,
            {'model': model, 'optimizer': optimizer})

        EarlyStoppingHandler = EarlyStopping(
            score_function=lambda engine: -engine.state.metrics['Loss'],
            trainer=trainer, patience=config['patience'])
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            EarlyStoppingHandler)

        trainer.run(TrainDataloader,
            max_epochs=config['n_epochs'], epoch_length=config['iters_per_epoch'])
        return outputdir

     
    def evaluate(
        self, 
        model_path: str, 
        is_image: bool = True,
        **kwargs):
        """
        Audio tagging on Audioset
        """

        config = torch.load(
            glob(os.path.join(model_path, '*config*'))[0], map_location='cpu')
        for k, v in kwargs.items():
            config[k] = v
        
        test = utils.parse_data(
            config['test'],
            debug=False,
            seed=self.seed
        )
        
        print(f"Number of test: {len(test)}")

        # _, eval_transform = utils.get_transform(
        #     hard_transform=False,
        #     is_image=is_image
        # )
        transform_kwargs = config.get('transform_kwargs', {})
        _, eval_transform = utils.get_transform(
            is_image=is_image,
            **transform_kwargs
        )
        TestDataloader = create_dataloader(
            input_h5=config['test'],
            indices=test,
            is_eval=True,
            is_image=is_image,
            transform=eval_transform,
            **config['dataloader_args']
        )

        model, _, _ = utils.get_model_from_pretrain(
            model_path=model_path,
            config=config,
            load_level='all',
            resume=True
        )
        model.cuda(0)

        def test():
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(TestDataloader, desc='Testing', ncols=80):
                    preds, labels = Runner._forward(model, batch)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
            all_preds, all_labels = torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
            all_preds = torch.sigmoid(all_preds).numpy()
            all_labels = all_labels.numpy()
            metrics = get_evaluate_metrics()(all_preds, all_labels)
            metrics = {**get_mlc_metrics(all_preds, all_labels), **metrics}
            for name, metric in metrics.items():
                print('{}: {:<5.3f}'.format(name, metric))

        test()

    def embedding(
        self, 
        output: str, 
        model_path: str,
        config: str = 'config/embedding.yaml', 
        is_image: bool = True,
        **kwargs):
        """
        Extract embeddings
        """

        config = utils.parse_config(config, debug=False, **kwargs)
        with h5py.File(config['train'], 'r') as train_input:
            n_class = train_input['target'].shape[-1]
            config['model_args']['n_class'] = n_class
        
        train = utils.parse_data(
            config['train'],
            debug=False,
            seed=self.seed,
        )
        # test = utils.parse_data(
        #     config['test'],
        #     debug=False,
        #     seed=self.seed,
        # )


        print(f"Number of train: {len(train)}")
        # print(f"Number of test: {len(test)}")

        _, transform = utils.get_transform(
            hard_transform=False,
            is_image=is_image
        )
        Dataloader = create_dataloader(
            input_h5=config['train'],
            indices=train,
            is_eval=True,
            transform=transform,
            is_image=is_image,
            seed=self.seed,
            **config['dataloader_args']
        )

        model, _, _ = utils.get_model_from_pretrain(
            model_path=model_path,
            config=config,
            resume=True
        )
        model.cuda()
        encoder = model.encoder.backbone
        encoder = encoder.eval()

        targets, embeddings = [], []
        with torch.no_grad():
            for data, target in tqdm(Dataloader, desc='Extracting', ncols=75):
                data = data[:, 0].cuda()
                embedding = encoder(data).cpu().numpy()
                embeddings.append(embedding)
                targets.append(target.numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        targets = np.concatenate(targets, axis=0)

        with h5py.File(output, 'w') as output:
            output.create_dataset('embeddings', shape=(
                (embeddings.shape[0], embeddings.shape[1])), dtype=np.float32)
            output.create_dataset('target', shape=(
                (targets.shape[0], targets.shape[1])), dtype=np.int16)
            output['embeddings'][:] = embeddings
            output['target'][:] = targets

if __name__ == '__main__':
    fire.Fire(Runner)
