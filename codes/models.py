import torch
import torch.nn as nn
import backbone as BackboneModel


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 1280,
        out_dim: int = 256,
        n_hidden: int = 1
    ):
        """
        The projection head for the encoder

        Parameters:
            in_dim (int): the input dimension
            out_dim (int): the output dimension
            n_hidden (int): the number of hidden layers
        """

        super(MLP, self).__init__()
        mlp = nn.ModuleList()
        _in_dim = in_dim
        for _ in range(n_hidden):
            next_dim = _in_dim // 2\
                if _in_dim >= 2 * out_dim else out_dim
            mlp.append(nn.Linear(_in_dim, next_dim))
            mlp.append(nn.BatchNorm1d(next_dim))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(p=0.1))
            _in_dim = next_dim
        mlp.append(nn.Linear(_in_dim, out_dim))
        
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        backbone_kwargs: dict = {},
        out_dim: int = 256,
        n_hidden: int = 1,
        if_pretrain: bool = True
    ):
        """
        Parameters:
            backbone (str): the name of the backbone model
            backbone_kwargs (dict): the keyword arguments for the backbone model
            out_dim (int): the output dimension of the encoder
            n_hidden (int): the number of hidden layers in the projection head
            is_pretrain (bool): whether the encoder is used for pretraining
        """

        super(Encoder, self).__init__()
        self.backbone = getattr(BackboneModel, backbone)(**backbone_kwargs)
        with torch.no_grad():
            if 'in_dim' in backbone_kwargs:
                x = torch.randn(4, backbone_kwargs['in_dim'])
            else:
                x = torch.randn(4, 3, 224, 224)
            rep_dim = self.backbone(x).shape[-1]
        self.rep_dim = rep_dim
        self.if_pretrain = if_pretrain
        if if_pretrain:
            self.proj_head = MLP(
                in_dim=rep_dim,
                out_dim=out_dim,
                n_hidden=n_hidden
            )

    def forward(self, x):
        rep = self.backbone(x)
        if not self.if_pretrain:
            return rep
        z = self.proj_head(rep)
        return z

    def load_params(self, params, only_backbone=True):
        if only_backbone:
            self.backbone.load_state_dict(params)
            print("Backbone load parameters")
        else:
            self.load_state_dict(params)
            print("Encoder load all parameters")
    
    def get_params(self, only_backbone=True):
        if only_backbone:
            return self.backbone.state_dict()
        else:
            return self.state_dict()

class Base(nn.Module):

    @torch.no_grad()
    def get_params(self, level='backbone'):
        if level in ['backbone', 'encoder']:
            return self.encoder.get_params(
                only_backbone=(level == 'backbone')
            )
        else:
            return self.state_dict()


class PretrainBase(Base):
    @torch.no_grad()
    def _set_key_encoder(self, encoder_q, encoder_k):
        for param_q, param_k in zip(
                encoder_q.parameters(), encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self, encoder_q, encoder_k):
        for param_q, param_k in zip(
                encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    @torch.no_grad()
    def _enqueue(
        self,
        MoCoQueue,
        LabelQueue,
        QueuePtr,
        QueueFull,
        k, labels
    ):
        B = len(labels)
        assert not self.cap_Q % B
        ptr = QueuePtr.item()
        MoCoQueue[..., ptr: ptr + B] = k.transpose(-2, -1)
        LabelQueue[ptr: ptr + B] = labels
        ptr = (ptr + B) % self.cap_Q
        QueuePtr[0] = ptr
        if not ptr: QueueFull[0] = True

    @torch.no_grad()
    def _load_params(self, state_dict, only_backbone=True):
        self.encoder_q.load_params(state_dict, only_backbone)
        self.encoder_k.load_params(state_dict, only_backbone)

class PretrainModel(PretrainBase):
    def __init__(
        self,
        backbone: str = 'resnet50',
        backbone_kwargs: dict = {},
        n_class: int = 80,
        output_func = lambda x: None,
        cap_Q: int = 2048,
        momentum: float = 0.999,
        out_dim: int = 256,
        n_hidden: int = 1,
        **kwargs
    ):
        
        """
        Parameters:
            backbone (str): the name of the backbone model
            backbone_kwargs (dict): the keyword arguments for the backbone model
            n_class (int): the number of classes
            output_func (function): the function to generate the output (all, any, MulSupCon)
            cap_Q (int): the capacity of the queue
            momentum (float): the momentum for the encoder_k update
            out_dim (int): the output dimension of the encoder
            n_hidden (int): the number of hidden layers in the projection head
        """

        super(PretrainModel, self).__init__()
        self.encoder_q = Encoder(
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
            out_dim=out_dim,
            n_hidden=n_hidden
        )
        self.encoder_k = Encoder(
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
            out_dim=out_dim,
            n_hidden=n_hidden
        )
        self.encoder = self.encoder_q
        self.register_buffer('QueuePtr',
            torch.tensor([0], dtype=torch.int32))
        self.register_buffer('QueueFull',
            torch.tensor([False], dtype=torch.bool))
        self.register_buffer('MoCoQueue',
            torch.randn(out_dim, cap_Q, dtype=torch.float32))
        self.register_buffer('LabelQueue',
            torch.zeros(cap_Q, n_class, dtype=torch.int64))
        self.MoCoQueue = nn.functional.normalize(self.MoCoQueue, dim=0)

        self.m, self.cap_Q, self.output_func = momentum, cap_Q, output_func
        self._set_key_encoder(self.encoder_q, self.encoder_k)


    def forward(self, data, labels):

        if self.training:
            self._momentum_update(self.encoder_q, self.encoder_k)
        q = self.encoder_q(data[:, 0])
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            k = self.encoder_k(data[:, 1])
            k = nn.functional.normalize(k, dim=1)

        queue_k = self.MoCoQueue.clone().detach()
        queue_labels = self.LabelQueue.clone().detach()
        if self.training:
            all_k = k
            all_labels = labels
            self._enqueue(
                self.MoCoQueue,
                self.LabelQueue,
                self.QueuePtr,
                self.QueueFull,
                all_k, all_labels
            )
        else:
            all_k = k
            all_labels = labels

        batch_score = torch.einsum(
            'bd,dk->bk', [q, all_k.transpose(0, 1)])
        if not self.QueueFull[0]:
            return self.output_func(
                labels, all_labels, batch_score)
        
        queue_score = torch.einsum(
            'bd,dk->bk', [q, queue_k])
        scores = torch.cat([batch_score, queue_score], dim=1)

        return self.output_func(
            labels, torch.cat([all_labels, queue_labels], dim=0), scores)

    def load_params(
        self,
        params,
        load_level: str = 'backbone'
    ):
        assert load_level in [
            'backbone',
            'encoder', # with proj_head
            'all'
        ]
        if load_level == 'backbone':
            self._load_params(params, only_backbone=True)
            print("Backbone load params")
        elif load_level == 'encoder':
            self._load_params(params, only_backbone=False)
            print("Encoder load all parameters")
        else:
            self.load_state_dict(params)
            print("Pretrain Model load all parameters")

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.encoder_q.backbone.parameters(), 'lr': lr * lrp},
            {'params': self.encoder_q.proj_head.parameters(), 'lr': lr}
        ]

class Simple(Base):
    def __init__(
        self,
        backbone,
        backbone_kwargs={},
        n_hidden=1,
        out_dim=256,
        n_class=80
    ):
        """
        Model to perform linear evaluation or finetuning

        Parameters:
            backbone (str): the name of the backbone model
            backbone_kwargs (dict): the keyword arguments for the backbone model
            n_hidden (int): the number of hidden layers in the projection head
            out_dim (int): the output dimension of the encoder
            n_class (int): the number of classes
        """
        
        super(Simple, self).__init__()
        self.encoder = Encoder(
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
            out_dim=out_dim,
            n_hidden=n_hidden,
            if_pretrain=False
        )
        with torch.no_grad():
            if 'in_dim' in backbone_kwargs:
                x = torch.randn(4, backbone_kwargs['in_dim'])
            else:
                x = torch.randn(4, 3, 224, 224)
            encoder_dim = self._forward_encoder(x).shape[-1]
        self.output_layer = nn.Sequential(
            nn.Linear(encoder_dim, n_class))
        self.linear_probe = False

    def _forward_encoder(self, data):
        embeddings = self.encoder(data)
        return embeddings
    
    def forward(self, data, labels):
        data = data[:, 0]
        if self.linear_probe:
            self.encoder.eval()
            with torch.no_grad():
                embeddings = self._forward_encoder(data)
        else:
            embeddings = self._forward_encoder(data)
        output = self.output_layer(embeddings)
        return output, labels
    
    def load_params(
            self,
            params,
            load_level: str = 'backbone'
    ):
        assert load_level in [
            'backbone',
            'encoder', # with proj
            'all'
        ]
        if load_level == 'all':
            self.load_state_dict(params)
            print("Model load all parameters")
        else:
            self.encoder.load_params(params, only_backbone=(load_level == 'backbone'))


    def extract_rep(self, data):
        return self.encoder(data)

    def set_linear_probe(self, linear_probe=True):
        if linear_probe:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.linear_probe = linear_probe

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.encoder.parameters(), 'lr': lr * lrp},
            {'params': self.output_layer.parameters(), 'lr': lr},
        ]
    
