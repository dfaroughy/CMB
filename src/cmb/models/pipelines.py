import torch
import torch.nn as nn
from dataclasses import dataclass

class Pipeline:
    def __init__(self, 
                 config: dataclass,
                 trained_model: nn.Module,
                 dynamics=None, 
                 best_epoch_model: bool=True
                 ):

        self.config = config
        self.model = trained_model.best_epoch_model if best_epoch_model else trained_model.last_epoch_model
        self.dynamics = dynamics
        self.time_steps = torch.linspace(0.0, 1.0 - config.pipeline.time_eps, config.pipeline.num_timesteps)

    @torch.no_grad()
    def generate_samples(self, **source):
        
        if self.config.pipeline.method == 'EulerSolver':
            solver = EulerSolver(config=self.config, model=self.model, dynamics=self.dynamics)
            paths = solver.simulate(time_steps=self.time_steps, **source)
            self.paths = paths.detach().cpu()

        elif self.config.pipeline.method == 'TauLeapingSolver':
            solver = TauLeapingSolver(config=self.config, model=self.model, dynamics=self.dynamics)
            jumps = solver.simulate(time_steps=self.time_steps, **source)
            self.jumps = jumps.detach().cpu()

        elif self.config.pipeline.method == 'EulerLeapingSolver':
            solver = EulerLeapingSolver(config=self.config, model=self.model, dynamics=self.dynamics)
            paths, jumps = solver.simulate(time_steps=self.time_steps, **source)
            self.paths = paths.detach().cpu()
            self.jumps = jumps.detach().cpu()

        elif self.config.pipeline.method == 'EulerMaruyamaLeapingSolver':
            solver = EulerMaruyamaLeapingSolver(config=self.config, model=self.model, dynamics=self.dynamics)
            paths, jumps = solver.simulate(time_steps=self.time_steps, **source)
            self.paths = paths.detach().cpu()
            self.jumps = jumps.detach().cpu()
        
        else:
            raise ValueError('Unknown pipeline method.')


class EulerSolver:
    ''' Euler ODE solver for continuous states
    '''
    def __init__(self, config, model, dynamics=None):
        self.device = config.train.device
        self.model = model.to(self.device)

    def simulate(self, 
                 time_steps, 
                 source_continuous, 
                 context_continuous=None, 
                 context_discrete=None,
                 mask=None):
        
        x = source_continuous.to(self.device)
        time_steps = time_steps.to(self.device)
        context_continuous = context_continuous.to(self.device) if context_continuous is not None else None
        context_discrete = context_discrete.to(self.device) if context_discrete is not None else None
        mask = mask.to(self.device) if mask is not None else None
        
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        paths = [x.clone()]

        for time in time_steps[1:]:
            time = torch.full((x.size(0), 1), time.item(), device=self.device)
            vector = self.model(t=time, 
                                x=x, 
                                context_continuous=context_continuous, 
                                context_discrete=context_discrete, 
                                mask=mask).to(self.device)
            x += delta_t * vector
            x *= mask
            paths.append(x.clone())
        
        paths = torch.stack(paths)

        return paths

class EulerMaruyamaSolver:
    ''' Euler-Maruyama SDE solver for continuous states
    '''
    def __init__(self, config, model, dynamics=None):
        self.device = config.train.device
        self.model = model.to(self.device)
        self.ref_bridge = dynamics.ref_bridge_continuous  

    def simulate(self, 
                 time_steps, 
                 source_continuous, 
                 context_continuous=None, 
                 context_discrete=None,
                 mask=None):
        
        x = source_continuous.to(self.device)
        time_steps = time_steps.to(self.device)
        context_continuous = context_continuous.to(self.device) if context_continuous is not None else None
        context_discrete = context_discrete.to(self.device) if context_discrete is not None else None
        mask = mask.to(self.device) if mask is not None else None
        
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        paths = [x.clone()]

        for time in time_steps[1:]:
            time = torch.full((x.size(0), 1), time.item(), device=self.device)
            drift = self.model(t=time, 
                               x=x, 
                               context_continuous=context_continuous, 
                               context_discrete=context_discrete, 
                               mask=mask).to(self.device)
            
            diffusion = self.ref_bridge.diffusion(t=delta_t).to(self.device)
            delta_w = torch.randn_like(x).to(self.device)
            x += delta_t * drift + diffusion * delta_w
            x *= mask

            paths.append(x.clone())
        
        paths = torch.stack(paths)

        return paths
    
    
class TauLeapingSolver:
    ''' Tau-Leaping solver for discrete states
    '''
    def __init__(self, config, model, dynamics=None):
        self.device = config.train.device
        self.dim_discrete = config.data.dim.features.discrete
        self.vocab_size = config.data.vocab_size.features 

        self.model = model.to(self.device)
        self.ref_bridge = dynamics.ref_bridge_discrete

    def simulate(self, 
                 time_steps, 
                 source_discrete, 
                 context_continuous=None, 
                 context_discrete=None,
                 mask=None, 
                 max_rate_last_step=False):
        
        k = source_discrete.to(self.device)
        time_steps = time_steps.to(self.device)
        context_continuous = context_continuous.to(self.device) if context_continuous is not None else None
        context_discrete = context_discrete.to(self.device) if context_discrete is not None else None
        mask = mask.to(self.device) if mask is not None else None

        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        jumps = [k.clone()]

        for time in time_steps[1:]:
            time = torch.full((k.size(0), 1), time.item(), device=self.device)
            
            logits = self.model(t=time, 
                                k=k, 
                                context_continuous=context_continuous, 
                                context_discrete=context_discrete , 
                                output_rates=True)
            
            rates = self.ref_bridge.rates(t=time, k=k, logits=logits).to(self.device)
    
            max_rate = torch.max(rates, dim=2)[1]
            all_jumps = torch.poisson(rates * delta_t).to(self.device) 
            _mask =  torch.sum(all_jumps, dim=-1).type_as(k) <= 1
            diff = torch.arange(self.vocab_size, device=self.device).view(1, 1, self.vocab_size) - k[:,:, None]
            net_jumps = torch.sum(all_jumps * diff, dim=-1).type_as(k)
            k += net_jumps * _mask
            k = torch.clamp(k, min=0, max=self.vocab_size-1)      
            k = (k.unsqueeze(-1) * mask).squeeze(-1)       
            jumps.append(k.clone())

        jumps = torch.stack(jumps)
        if max_rate_last_step:
            jumps[-1] = max_rate # replace last jump with max rates

        return jumps

class EulerLeapingSolver:
    ''' EulerLeaping solver for hybrid states combining Euler (ODE) and Tau-Leaping steps.
    '''
    def __init__(self, config, model, dynamics=None):
        self.device = config.train.device
        self.dim_discrete = config.data.dim.features.discrete
        self.vocab_size = config.data.vocab_size.features 

        self.model = model.to(self.device)
        self.ref_bridge = dynamics.ref_bridge_discrete

    def simulate(self, 
                 time_steps, 
                 source_continuous, 
                 source_discrete, 
                 context_continuous=None, 
                 context_discrete=None, 
                 mask=None, 
                 max_rate_last_step=False):
        
        x = source_continuous.to(self.device)
        k = source_discrete.to(self.device)
        time_steps = time_steps.to(self.device)
        context_continuous = context_continuous.to(self.device) if context_continuous is not None else None
        context_discrete = context_discrete.to(self.device) if context_discrete is not None else None
        mask = mask.to(self.device) if mask is not None else None

        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        paths, jumps = [x.clone()], [k.clone()]

        for time in time_steps[1:]:

            time = torch.full((x.size(0), 1), time.item(), device=self.device)

            #...compute velocity and rates:
            vector, logits = self.model(t=time, 
                                       x=x, 
                                       k=k, 
                                       context_continuous=context_continuous, 
                                       context_discrete=context_discrete,
                                       mask=mask)
            
            vector = vector.to(self.device)
            rates = self.ref_bridge.rates(t=time, k=k, logits=logits).to(self.device)

            #...tau-leaping step:
            max_rate = torch.max(rates, dim=2)[1]
            all_jumps = torch.poisson(rates * delta_t).to(self.device) 
            jump_mask =  torch.sum(all_jumps, dim=-1).type_as(k) <= 1
            diff = torch.arange(self.vocab_size, device=self.device).view(1, 1, self.vocab_size) - k[:,:, None]
            net_jumps = torch.sum(all_jumps * diff, dim=-1).type_as(k)
            k += net_jumps * jump_mask
            k = torch.clamp(k, min=0, max=self.vocab_size-1)    
            k = (k.unsqueeze(-1) * mask).squeeze(-1)       
            jumps.append(k.clone())

            #...euler step:
            x += delta_t * vector
            x *= mask
            paths.append(x.clone())
        
        paths = torch.stack(paths)
        jumps = torch.stack(jumps)

        if max_rate_last_step:
            jumps[-1] = max_rate # replace last jump with max rates

        return paths, jumps
    

class EulerMaruyamaLeapingSolver:
    ''' Euler-Maruyama-Leaping solver for hybrid states combining Euler-Maruyama (SDE) and Tau-Leaping steps
    '''
    def __init__(self, config, model, dynamics=None):
        self.device = config.train.device
        self.dim_discrete = config.data.dim.features.discrete
        self.vocab_size = config.data.vocab_size.features 

        self.model = model.to(self.device)
        self.ref_bridge_continuous = dynamics.ref_bridge_continuous
        self.ref_bridge_discrete = dynamics.ref_bridge_discrete

    def simulate(self, 
                 time_steps, 
                 source_continuous, 
                 source_discrete, 
                 context_continuous=None, 
                 context_discrete=None, 
                 mask=None, 
                 max_rate_last_step=False):
        
        x = source_continuous.to(self.device)
        k = source_discrete.to(self.device)
        time_steps = time_steps.to(self.device)
        context_continuous = context_continuous.to(self.device) if context_continuous is not None else None
        context_discrete = context_discrete.to(self.device) if context_discrete is not None else None
        mask = mask.to(self.device) if mask is not None else None

        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        paths, jumps = [x.clone()], [k.clone()]

        for time in time_steps[1:]:

            time = torch.full((x.size(0), 1), time.item(), device=self.device)

            #...compute velocity and rates:
            drift, logits = self.model(t=time, 
                                       x=x, 
                                       k=k, 
                                       context_continuous=context_continuous, 
                                       context_discrete=context_discrete)
            
            drift = drift.to(self.device)
            diffusion = self.ref_bridge_continuous.diffusion(t=time)#.to(self.device)
            rates = self.ref_bridge_discrete.rates(t=time, k=k, logits=logits).to(self.device)

            #...tau-leaping step:
            max_rate = torch.max(rates, dim=2)[1]
            all_jumps = torch.poisson(rates * delta_t).to(self.device) 
            _mask =  torch.sum(all_jumps, dim=-1).type_as(k) <= 1
            diff = torch.arange(self.vocab_size, device=self.device).view(1, 1, self.vocab_size) - k[:,:, None]
            net_jumps = torch.sum(all_jumps * diff, dim=-1).type_as(k)
            k += net_jumps * _mask
            k = torch.clamp(k, min=0, max=self.vocab_size-1)     
            k = (k.unsqueeze(-1) * mask).squeeze(-1)       
            jumps.append(k.clone())

            #...euler-maruyama step:
            noise = torch.randn_like(x).to(self.device)
            delta_w = torch.sqrt(delta_t) * noise
            x += delta_t * drift + diffusion * delta_w
            x *= mask
            paths.append(x.clone())
        
        paths = torch.stack(paths)
        jumps = torch.stack(jumps)

        if max_rate_last_step:
            jumps[-1] = max_rate # replace last jump with max rates

        return paths, jumps
