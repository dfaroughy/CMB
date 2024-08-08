import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass, fields

#----------------------------------------------
# utils for trainers           
#----------------------------------------------

class Train_Step(nn.Module): 

    """ Represents a training step.
    """

    def __init__(self):
        super(Train_Step, self).__init__()
        self.loss = 0
        self.epoch = 0
        self.losses = []

    def update(self, model, loss_fn, dataloader: DataLoader, optimizer):
        self.loss = 0
        self.epoch += 1
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            loss_current = loss_fn(model, batch)
            loss_current.backward()
            optimizer.step()  
            self.loss += loss_current.detach().cpu().numpy() / len(dataloader)
        self.losses.append(self.loss) 

class Validation_Step(nn.Module):

    """ Represents a validation step.
    """

    def __init__(self):
        super(Validation_Step, self).__init__()
        self.loss = 0
        self.epoch = 0
        self.patience = 0
        self.loss_min = np.inf
        self.losses = []
        
    @torch.no_grad()
    def update(self, model, loss_fn, dataloader: DataLoader):
        self.epoch += 1
        self.loss = 0
        self.validate = bool(dataloader)
        if self.validate:
            model.eval()
            for batch in dataloader:
                loss_current = loss_fn(model, batch)
                self.loss += loss_current.detach().cpu().numpy() / len(dataloader)
            self.losses.append(self.loss) 

    @torch.no_grad()
    def checkpoint(self, min_epochs, early_stopping=None):
        terminate = False
        improved = False
        if self.validate:
            if self.loss < self.loss_min:
                self.loss_min = self.loss
                self.patience = 0
                improved = True 
            else: self.patience += 1 if self.epoch > min_epochs else 0
            if self.patience >= early_stopping: terminate = True
        return terminate, improved


class Optimizer:

    """
    Custom optimizer class with support for gradient clipping.
    
    Attributes:
    - configs: Configuration dataclass containing optimizer configurations.
    """

    def __init__(self, config: dataclass):
        self.config = config
        self.optimizer = config.OPTIMIZER
        self.lr = config.LR 
        self.weight_decay = config.WEIGHT_DECAY
        self.betas = config.OPTIMIZER_BETAS
        self.eps = config.OPTIMIZER_EPS
        self.amsgrad = config.OPTIMIZER_AMSGRAD
        self.gradient_clip = config.GRADIENT_CLIP 

    def get_optimizer(self, parameters):

        optim_args = {'lr': self.lr, 'weight_decay': self.weight_decay}

        if self.optimizer == 'Adam':
            if hasattr(self.config, 'betas'): optim_args['betas'] = self.betas
            if hasattr(self.config, 'eps'): optim_args['eps'] = self.eps
            if hasattr(self.config, 'amsgrad'): optim_args['amsgrad'] = self.amsgrad
            return torch.optim.Adam(parameters, **optim_args)
        
        elif self.optimizer == 'AdamW':
            if hasattr(self.config, 'betas'): optim_args['betas'] = self.betas
            if hasattr(self.config, 'eps'): optim_args['eps'] = self.eps
            if hasattr(self.config, 'amsgrad'): optim_args['amsgrad'] = self.amsgrad
            return torch.optim.AdamW(parameters, **optim_args)
        
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
    def clip_gradients(self, optimizer):
        if self.gradient_clip: torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], self.gradient_clip)

    def __call__(self, parameters):
        optimizer = self.get_optimizer(parameters)
        #...override the optimizer.step() to include gradient clipping
        original_step = optimizer.step

        def step_with_clipping(closure=None):
            self.clip_gradients(optimizer)
            original_step(closure)          
        optimizer.step = step_with_clipping
        return optimizer

class Scheduler:

    """
    Custom scheduler class to adjust the learning rate during training.
    
    Attributes:
    - configs: Configuration dataclass containing scheduler configurations.
    """

    def __init__(self, config: dataclass):
        self.scheduler = config.SCHEDULER
        self.T_max = config.SCHEDULER_T_MAX
        self.eta_min = config.SCHEDULER_ETA_MIN
        self.gamma = config.SCHEDULER_GAMMA
        self.step_size = config.SCHEDULER_STEP_SIZE

    def get_scheduler(self, optimizer):
        if self.scheduler == 'CosineAnnealingLR': return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)
        elif self.scheduler == 'StepLR': return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        elif self.scheduler == 'ExponentialLR': return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        elif self.scheduler is None: return NoScheduler(optimizer)
        else: raise ValueError(f"Unsupported scheduler: {self.scheduler}")

    def __call__(self, optimizer):
        return self.get_scheduler(optimizer)

class NoScheduler:
    def __init__(self, optimizer): pass
    def step(self): pass    


class Logger:
    ''' Logging handler for training and validation.
    '''
    def __init__(self, configs: dataclass, path: Path):
        self.path = path
        self.fh = None  
        self.ch = None 
        self._training_loggers()

    def _training_loggers(self):
        
        self.logfile = logging.getLogger('file_logger')
        self.logfile.setLevel(logging.INFO)
        self.fh = logging.FileHandler(self.path) 
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.logfile.addHandler(self.fh)
        self.logfile.propagate = False 
        
        self.console = logging.getLogger('console_logger')
        self.console.setLevel(logging.INFO)
        self.ch = logging.StreamHandler()  
        ch_formatter = logging.Formatter('%(message)s') 
        self.ch.setFormatter(ch_formatter)
        self.console.addHandler(self.ch)
        self.console.propagate = False 

    def logfile_and_console(self, message):
        self.logfile.info(message)
        self.console.info(message)

    def close(self):
        if self.fh:
            self.fh.close()
            self.logfile.removeHandler(self.fh)
        if self.ch:
            self.ch.close()
            self.console.removeHandler(self.ch)


#----------------------------------------------
# utils for pipelines            
#----------------------------------------------

class ContextWrapper(torch.nn.Module):
    """ Wraps model to torchdyn compatible format.
    """
    def __init__(self, net, context=None):
        super().__init__()
        self.nn = net
        self.context = context

    def forward(self, t, x):
        t = t.repeat(x.shape[0])
        t = self.reshape_time_like(t, x)
        return self.nn(t=t, x=x, context=self.context)

    def reshape_time_like(self, t, x):
        if isinstance(t, (float, int)): return t
        else: return t.reshape(-1, *([1] * (x.dim() - 1)))


class EulerSolver:
    def __init__(self, vector_field, device):
        self.vector_field = vector_field
        self.device = device

    def trajectory(self, x, t_span):
        time_steps = len(t_span)
        dt = (t_span[-1] - t_span[0]) / (time_steps - 1)
        trajectory = [x]

        for i in range(1, time_steps):
            t = t_span[i-1]
            x = x + dt * self.vector_field(t, x).to(self.device)
            trajectory.append(x)

        return torch.stack(trajectory)



import torch
from dataclasses import dataclass
from torch.nn.functional import softmax

class TransitionRateModel(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model # model should output logits
        self.vocab_size = config.VOCAB_SIZE
        self.config = config
        self.gamma = config.GAMMA
        self.time_epsilon = config.TIME_EPS

    def forward(self, t, s, context=None):
        t = t.squeeze()
        if len(s.shape) != 2:
            s = s.reshape(s.size(0),-1)
        logits = self.model(t, s, context)
        t1 = 1. - self.time_epsilon
        beta_integral = (t1 - t) * self.gamma
        wt = torch.exp(-self.vocab_size * beta_integral)
        A, B, C = 1. , (wt * self.vocab_size)/(1. - wt) , wt
        qx = softmax(logits, dim=2)
        qy = torch.gather(qx, 2, s.long().unsqueeze(2))
        rate = A + B[:, None, None] * qx + C[:, None, None] * qy
        return rate

class TauLeapingSolver:
    def __init__(self, transition_rate, device):
        self.transition_rate = transition_rate
        self.device = device

    def simulate(self, x, t_span):
        time_steps = len(t_span)
        tau = (t_span[-1] - t_span[0]) / (time_steps - 1)
        trajectory = [x]

        for i in range(1, time_steps):
            t = t_span[i-1]
    
            current_state = x.clone()
            rates = self.transition_rate(t, current_state).to(self.device)
            voc_size = rates.size(-1) 

            jumps = torch.poisson(rates * tau).to(self.device)  
            net_jumps = torch.argmax(jumps, dim=-1).type_as(current_state)
            x = torch.clamp(net_jumps, min=0, max=voc_size-1)            
            trajectory.append(x.clone())

        return torch.stack(trajectory)