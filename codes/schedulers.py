from torch.optim.lr_scheduler import\
    CosineAnnealingWarmRestarts,\
    CosineAnnealingLR,\
    ReduceLROnPlateau

class CosineAnnealingLRReduce():
    def __init__(
        self,
        optimizer,
        T_max,
        eta_min,
        LR_mult,
        state_dict = None,
        warmup: bool = False,
        warmup_from: float = 0.01,
        warm_epochs: int = 10,
        **kwargs
    ):
        assert LR_mult <= 1
        self.LR_mult = LR_mult
        self.T_max = T_max
        self.eta_min = eta_min
        self.optimizer = optimizer
        self.base_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min,
        )
        if state_dict:
            self.load_state_dict(state_dict)
        self.warmup = warmup
        self.warmup_from = warmup_from
        self.warm_epochs = warm_epochs
        if warmup:
            self.warmup_to = optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_from
            self.epoch = 1
        

    def step(self):
        if self.warmup and self.epoch <= self.warm_epochs:
            # warm up
            lr = self.warmup_from + (self.warmup_to - self.warmup_from) * self.epoch / self.warm_epochs
            self.epoch += 1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return
        if self.base_scheduler.last_epoch % self.T_max:
            self.base_scheduler.step()
            return
        if (self.base_scheduler.last_epoch // self.T_max) % 2:
            for idx in range(len(self.base_scheduler.base_lrs)):
                self.base_scheduler.base_lrs[idx] *= self.LR_mult
        if (not (self.base_scheduler.last_epoch // self.T_max) % 2) and self.base_scheduler.last_epoch:
            self.base_scheduler.eta_min *= self.LR_mult
        self.base_scheduler.step()

    def get_lr(self):
        return self.base_scheduler.optimizer.param_groups[0]['lr'] 
    
    def state_dict(self):
        return {
            'wrapper': {
                'LR_mult': self.LR_mult,
                'T_max': self.T_max,
                'eta_min': self.eta_min
            },
            'base': self.base_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        wrapper_state_dict = state_dict['wrapper']
        base_state_dict = state_dict['base']
        self.LR_mult = wrapper_state_dict['LR_mult']
        self.T_max = wrapper_state_dict['T_max']
        self.eta_min = wrapper_state_dict['eta_min']
        self.base_scheduler.load_state_dict(base_state_dict)
        
class CosineAnnealingRestartLRReduce():
    def __init__(self, optimizer, T_0, T_mult, eta_min, LR_mult):
        assert T_mult >= 1
        assert LR_mult <= 1
        self.base_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        self.LR_mult = LR_mult

    def step(self):
        if self.base_scheduler.T_cur >= self.base_scheduler.T_i - 1:
            for idx in range(len(self.base_scheduler.base_lrs)):
                self.base_scheduler.base_lrs[idx] *= self.LR_mult
                self.base_scheduler.eta_min *= self.LR_mult
        self.base_scheduler.step()

    def get_lr(self):
        return self.base_scheduler.optimizer.param_groups[0]['lr']

CosineAnnealingWarmRestarts = CosineAnnealingWarmRestarts
CosineAnnealingLR = CosineAnnealingLR
ReduceLROnPlateau = ReduceLROnPlateau
