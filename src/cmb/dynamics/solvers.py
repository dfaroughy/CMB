import torch
import torch.nn as nn
from dataclasses import dataclass

class EulerSolver:
    ''' Euler ODE solver for continuous states
    '''
    def __init__(self, config, dynamics_discrete=None, dynamics_continuous=None):
        self.device = config.train.device

    def simulate(self, 
                 model,
                 time_steps, 
                 source_continuous, 
                 context_continuous=None, 
                 context_discrete=None,
                 output_history=False,
                 mask=None):
        
        model = model.to(self.device)
        x = source_continuous.to(self.device)
        time_steps = time_steps.to(self.device)
        context_continuous = context_continuous.to(self.device) if context_continuous is not None else None
        context_discrete = context_discrete.to(self.device) if context_discrete is not None else None
        mask = mask.to(self.device) if mask is not None else None
        
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        trajectories = x.clone() if output_history else [x.clone()]

        for time in time_steps[1:]:
            time = torch.full((x.size(0), 1), time.item(), device=self.device)
            
            vector, _ = model(t=time, 
                              x=x, 
                              context_continuous=context_continuous, 
                              context_discrete=context_discrete, 
                              mask=mask)
            
            vector = vector.to(self.device)
            x += delta_t * vector
            x *= mask

            if output_history: trajectories.append(x.clone())
            else: trajectories = x.clone()
        
        if output_history:
            trajectories = torch.stack(trajectories)

        return trajectories.detach().cpu(), None

class EulerMaruyamaSolver:
    ''' Euler-Maruyama SDE solver for continuous states
    '''
    def __init__(self, config, dynamics_continuous, dynamics_discrete=None):
        self.device = config.train.device
        self.diffusion = dynamics_continuous.diffusion 

    def simulate(self, 
                 model,
                 time_steps, 
                 source_continuous, 
                 context_continuous=None, 
                 context_discrete=None,
                 output_history=False,
                 mask=None):
        
        model = model.to(self.device)
        time_steps = time_steps.to(self.device)
        x = source_continuous.to(self.device)
        context_continuous = context_continuous.to(self.device) if context_continuous is not None else None
        context_discrete = context_discrete.to(self.device) if context_discrete is not None else None
        mask = mask.to(self.device) if mask is not None else None
        
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        trajectories = x.clone() if output_history else [x.clone()]

        for time in time_steps[1:]:
            time = torch.full((x.size(0), 1), time.item(), device=self.device)
            
            drift, _ = model(t=time, 
                             x=x, 
                             context_continuous=context_continuous, 
                             context_discrete=context_discrete, 
                             mask=mask)
            
            drift = drift.to(self.device)
            diffusion = self.diffusion(t=delta_t).to(self.device)
            delta_w = torch.randn_like(x).to(self.device)
            x += delta_t * drift + diffusion * delta_w
            x *= mask

            if output_history: trajectories.append(x.clone())
            else: trajectories = x.clone()
        
        if output_history:
            trajectories = torch.stack(trajectories)        

        return trajectories.detach().cpu(), None
    
    
class TauLeapingSolver:
    ''' Tau-Leaping solver for discrete states
    '''
    def __init__(self, config, dynamics_discrete, dynamics_continuous=None):
        self.device = config.train.device
        self.dim_discrete = config.data.dim.features.discrete
        self.vocab_size = config.data.vocab.size.features 
        self.rate = dynamics_discrete.rate

    def simulate(self, 
                 model,
                 time_steps, 
                 source_discrete, 
                 context_continuous=None, 
                 context_discrete=None,
                 mask=None, 
                 max_rate_last_step=False):
        
        model = model.to(self.device)
        time_steps = time_steps.to(self.device)
        k = source_discrete.to(self.device)
        context_continuous = context_continuous.to(self.device) if context_continuous is not None else None
        context_discrete = context_discrete.to(self.device) if context_discrete is not None else None
        mask = mask.to(self.device) if mask is not None else None

        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        jumps = [k.clone()]

        for time in time_steps[1:]:
            time = torch.full((k.size(0), 1), time.item(), device=self.device)
            
            logits = model(t=time, 
                            k=k, 
                            context_continuous=context_continuous, 
                            context_discrete=context_discrete, 
                            output_rates=True)
            
            rates = self.rate(t=time, k=k, logits=logits).to(self.device)
    
            max_rate = torch.max(rates, dim=2)[1]
            all_jumps = torch.poisson(rates * delta_t).to(self.device) 
            _mask =  torch.sum(all_jumps, dim=-1).type_as(k.squeeze(-1)) <= 1
            diff = torch.arange(self.vocab_size, device=self.device).view(1, 1, self.vocab_size) - k[:,:, None]
            net_jumps = torch.sum(all_jumps * diff, dim=-1).type_as(k)
            k += net_jumps * _mask
            k = torch.clamp(k, min=0, max=self.vocab_size-1)      
            k = (k.unsqueeze(-1) * mask).squeeze(-1)       
            jumps.append(k.clone())

        jumps = torch.stack(jumps)
        if max_rate_last_step:
            jumps[-1] = max_rate.unsqueeze(-1) # replace last jump with max rates

        return None, jumps.detach().cpu()

class EulerLeapingSolver:
    ''' EulerLeaping solver for hybrid states combining Euler (ODE) and Tau-Leaping steps.
    '''
    def __init__(self, config, dynamics_discrete, dynamics_continuous=None):
        self.device = config.train.device
        self.dim_discrete = config.data.dim.features.discrete
        self.vocab_size = config.data.vocab.size.features 
        self.rate = dynamics_discrete.rate

    def simulate(self, 
                 model,
                 time_steps, 
                 source_continuous, 
                 source_discrete, 
                 context_continuous=None, 
                 context_discrete=None, 
                 mask=None, 
                 output_history=False,
                 max_rate_last_step=False):
        
        model = model.to(self.device)
        time_steps = time_steps.to(self.device)
        x = source_continuous.to(self.device)
        k = source_discrete.to(self.device)
        context_continuous = context_continuous.to(self.device) if context_continuous is not None else None
        context_discrete = context_discrete.to(self.device) if context_discrete is not None else None
        mask = mask.to(self.device) if mask is not None else None

        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        trajectories = [x.clone()] if output_history else x.clone()
        jumps = [k.clone()] if output_history else k.clone()

        for time in time_steps[1:]:

            time = torch.full((x.size(0), 1), time.item(), device=self.device)

            #...compute velocity and rates:
            
            vector, logits = model(t=time, 
                                    x=x, 
                                    k=k, 
                                    context_continuous=context_continuous, 
                                    context_discrete=context_discrete,
                                    mask=mask)
            
            vector = vector.to(self.device)
            rates = self.rate(t=time, k=k, logits=logits).to(self.device)

            #...tau-leaping step:
            k = k.squeeze(-1)
            max_rate = torch.max(rates, dim=2)[1]
            all_jumps = torch.poisson(rates * delta_t).to(self.device) 
            jump_mask =  torch.sum(all_jumps, dim=-1).type_as(k) <= 1
            diff = torch.arange(self.vocab_size, device=self.device).view(1, 1, self.vocab_size) - k[:, :, None]
            net_jumps = torch.sum(all_jumps * diff, dim=-1).type_as(k)
            k += net_jumps * jump_mask
            k = torch.clamp(k, min=0, max=self.vocab_size-1)    
            k = k.unsqueeze(-1)
            k *= mask

            #...euler step:
            x += delta_t * vector
            x *= mask

            if output_history:
                trajectories.append(x.clone())
                jumps.append(k.clone())
            else:
                trajectories, jumps = x.clone(), k.clone()

        if output_history:
            trajectories = torch.stack(trajectories)
            if max_rate_last_step:
                jumps[-1] = max_rate # replace last jump with max rates
            jumps = torch.stack(jumps)

        if max_rate_last_step:
            jumps = max_rate.unsqueeze(-1) # replace last jump with max rates
            
        return trajectories.detach().cpu(), jumps.detach().cpu()
    

class EulerMaruyamaLeapingSolver:
    ''' Euler-Maruyama-Leaping solver for hybrid states combining Euler-Maruyama (SDE) and Tau-Leaping steps
    '''
    def __init__(self, config, dynamics_discrete, dynamics_continuous):
        self.device = config.train.device
        self.dim_discrete = config.data.dim.features.discrete
        self.vocab_size = config.data.vocab.size.features 
        self.diffusion = dynamics_continuous.diffusion
        self.rate = dynamics_discrete.rate

    def simulate(self,
                 model,
                 time_steps, 
                 source_continuous, 
                 source_discrete, 
                 context_continuous=None, 
                 context_discrete=None, 
                 mask=None, 
                 max_rate_last_step=False,
                 output_history=False):
        
        model = model.to(self.device)
        time_steps = time_steps.to(self.device)
        x = source_continuous.to(self.device)
        k = source_discrete.to(self.device)
        context_continuous = context_continuous.to(self.device) if context_continuous is not None else None
        context_discrete = context_discrete.to(self.device) if context_discrete is not None else None
        mask = mask.to(self.device) if mask is not None else None

        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        trajectories, jumps = ([x.clone()], [k.clone()]) if output_history else x.clone(), k.clone()

        for time in time_steps[1:]:

            time = torch.full((x.size(0), 1), time.item(), device=self.device)

            #...compute velocity and rates:
            drift, logits = model(t=time, 
                                  x=x, 
                                  k=k, 
                                  context_continuous=context_continuous, 
                                  context_discrete=context_discrete)
            
            drift = drift.to(self.device)
            diffusion = self.diffusion(t=time) #.to(self.device)
            rates = self.rate(t=time, k=k, logits=logits).to(self.device)

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
            trajectories.append(x.clone())
        
            if output_history:
                trajectories.append(x.clone())
                jumps.append(k.clone())
            else:
                trajectories, jumps = x.clone(), k.clone()

        if output_history:
            trajectories = torch.stack(trajectories)
            if max_rate_last_step:
                jumps[-1] = max_rate # replace last jump with max rates
            jumps = torch.stack(jumps)

        if max_rate_last_step:
            jumps = max_rate.unsqueeze(-1) # replace last jump with max rates
            
        return trajectories.detach().cpu(), jumps.detach().cpu()
