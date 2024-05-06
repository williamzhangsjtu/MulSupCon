import torch
import numpy as np
from glob import glob
import h5py 
import yaml
import torch
import sys, os
import models
from loguru import logger
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

def parse_data(
    h5: str,
    debug: bool = False,
    seed: int = 0
):
    """
    get indices from h5 file
    """
    with h5py.File(h5, 'r') as input:
        indices = list(range(input['target'].shape[0]))
    random_state = np.random.RandomState(seed=seed)
    random_state.shuffle(indices)
    indices = indices[:int(len(indices) * 0.1)] if debug else indices
    return indices


def split_data(indices, seed=0, ratio=0):
    assert 0 <= ratio < 1, 'ratio must be in [0, 1)'
    if not ratio:
        return indices, []
    random_state = np.random.RandomState(seed=seed)
    random_state.shuffle(indices)
    size = int(len(indices) * (1 - ratio))
    return indices[:size], indices[size:]

def parse_config(config_file, debug=False, **kwargs):
    with open(config_file) as con_read:
        config = yaml.load(con_read, Loader=yaml.FullLoader)
    for k, v in kwargs.items():
        config[k] = v
    if debug:
        config['outputdir'] = 'experiment/debug'
        config['dataloader_args']['batch_size'] = 32
        config['dataloader_args']['num_workers'] = 4
        config['iters_per_epoch'] = 200
        if 'cap_Q' in config['model_args']:
            config['model_args']['cap_Q'] = 1024
    return config

# how to generate mask and score
def get_output_func(pattern='MulSupCon', with_weight=False):
    """
    Paremeters
        pattern: how to generate mask and score
            - all: labels exactly matched with the anchor
            - any: labels with at least one common label with the anchor
            - MulSupCon: treat each of anchor's label separately
        with_weight: argument for sep pattern, whether to use 1/|y| to weight the loss
    """
    assert pattern in ['all', 'any', 'MulSupCon']

    def generate_output_MulSupCon(batch_labels, ref_labels, scores):
        """
        MulSupCon
        
        Parameters:
            batch_labels: B x C tensor, labels of the anchor
            ref_labels: Q x C tensor, labels of samples from queue
            scores: B x Q tensor, cosine similarity between the anchor and samples from queue
        """
        B = len(batch_labels)
        indices = torch.where(batch_labels == 1)
        scores = scores[indices[0]]
        labels = torch.zeros(len(scores), batch_labels.shape[1], device=scores.device)
        labels[range(len(labels)), indices[1]] = 1
        masks = (labels @ ref_labels.T).to(torch.bool)
        n_score_per_sample = batch_labels.sum(dim=1).to(torch.int16).tolist()
        if with_weight:
            weights_per_sample = [1/(n * B) for n in n_score_per_sample for _ in range(n)]
        else:
            weights_per_sample = [1 / len(scores) for n in n_score_per_sample for _ in range(n)]
        weights_per_sample = torch.tensor(
            weights_per_sample,
            device=scores.device,
            dtype=torch.float32
        )
        return scores, [masks.to(torch.long), weights_per_sample]
    
    
    def generate_output_all(batch_label, ref_label, scores):
        """
        positives: labels exactly matched with the anchor
        """
        mul_matrix = (batch_label @ ref_label.T).to(torch.int16)
        mask1 = torch.sum(batch_label, dim=1).unsqueeze(1).to(torch.int16) == mul_matrix
        mask2 = torch.sum(ref_label, dim=1).unsqueeze(1).to(torch.int16) == mul_matrix.T
        mask = mask1 & mask2.T
        return scores, mask.to(torch.long)

    def generate_output_any(batch_label, ref_label, scores):
        """
        positives: labels with at least one common label with the anchor
        """
        mul_matrix = (batch_label @ ref_label.T)
        return scores, (mul_matrix > 0).to(torch.long)

    if pattern == 'all':
        return generate_output_all
    elif pattern == 'any':
        return generate_output_any
    elif pattern == 'MulSupCon':
        return generate_output_MulSupCon
    else:
        raise NotImplementedError
    

def get_model_from_pretrain(
    model_path: str,
    config: dict,
    resume: bool = False,
    load_level: str = 'backbone',
    **kwargs
):
    """
    Load model, optimizer, and scheduler from saved model
    
    """

    if model_path is None:
        model = getattr(models, config['model'])(**config['model_args'], **kwargs)
        return model, {}, {}
    pretrain_config = torch.load(
        glob(os.path.join(model_path, '*config*'))[0], map_location='cpu')
    if resume:
        config = pretrain_config
    model = getattr(models, config['model'])(**config['model_args'], **kwargs)
    saved = torch.load(
        glob(os.path.join(model_path, '*best*'))[0], map_location='cpu')
    params, optim_params, scheduler_params\
         = saved['model'], saved.get('optimizer', {}), saved.get('scheduler', {})
    pretrain_model = getattr(models, pretrain_config['model'])(**pretrain_config['model_args'])
    pretrain_model.load_state_dict(params)
    encoder_params = pretrain_model.get_params(level=load_level)
    model.load_params(encoder_params, load_level=load_level)
    return model, optim_params, scheduler_params


def genlogger(file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if file:
        logger.add(file, enqueue=True, format=log_format)
    return logger
    
class Logger():
    def __init__(self, file, rank=0):
        self.logger = None
        self.rank = rank
        if not rank:
            self.logger = genlogger(file)
    def info(self, msg):
        if not self.rank:
            self.logger.info(msg)


def get_transform(
    is_image: bool = True,
    h5: str = None,
    p: float = 0,
    **kwargs
):
    """
    Get transform function for data augmentation

    Parameters:
        is_image: whether the data is image
        h5: path to h5 file
        p: probability of masking
    """

    if not is_image:
        _scaler_transform = lambda x: x
        if h5:
            scaler = StandardScaler()
            with h5py.File(h5, 'r') as input:
                features = input['feature'][:]
            scaler.fit(features)
            _scaler_transform = scaler.transform
        scaler_transform = lambda x: torch.from_numpy(_scaler_transform(x))
        def mask_transform(vector):
            if not p:
                return vector
            mask = torch.zeros_like(vector)
            mask.uniform_()
            mask = (mask > p).to(torch.float32)
            return vector * mask
        train_transform = lambda x: mask_transform(scaler_transform(x))
        eval_transform = lambda x: scaler_transform(x)
        return train_transform, eval_transform
        
    
    resolution = kwargs.get('resolution', 224)
    strategy = kwargs.get('strategy', 'Simple')
    if strategy == 'SimCLR':
        print("Using SimCLR augmentation")
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    else: # trivial
        print("Using Simple augmentation")
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    return train_transform, eval_transform