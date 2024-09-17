import torch
from dataclasses import dataclass

class Pipeline:
    def __init__(self, 
                 trained_model, 
                 config: dataclass=None,
                 best_epoch_model: bool=True
                 ):

        self.config = config
        self.model = trained_model.best_epoch_model if best_epoch_model else trained_model.last_epoch_model
        self.time_steps = torch.linspace(0.0, 1.0 - config.time_eps, config.num_timesteps)

    @torch.no_grad()
    def generate_samples(self, **source):
        
        if self.config.sampler == 'EulerSolver':
            solver = EulerSolver(model=self.model, config=self.config)
            paths = solver.simulate(time_steps=self.time_steps, **source)
            self.paths = paths.detach().cpu()

        if self.config.sampler == 'EulerLeapingSolver':
            solver = EulerLeapingSolver(model=self.model, config=self.config)
            paths, jumps = solver.simulate(time_steps=self.time_steps, **source)
            self.paths = paths.detach().cpu()
            self.jumps = jumps.detach().cpu()

        if self.config.sampler == 'TauLeapingSolver':
            solver = TauLeapingSolver(model=self.model, config=self.config)
            jumps = solver.simulate(time_steps=self.time_steps, **source)
            self.jumps = jumps.detach().cpu()


class EulerSolver:
    def __init__(self, model, config):
        self.model = model # velocity field
        self.device = config.device

    def simulate(self, 
                 time_steps, 
                 source_continuous, 
                 context=None, 
                 mask=None):
        
        x = source_continuous.to(self.device)
        time_steps = time_steps.to(self.device)
        context = context.to(self.device) if context is not None else None
        mask = mask.to(self.device) if mask is not None else None
        
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        paths = [x.clone()]

        for time in time_steps[1:]:
            time = torch.full((x.size(0), 1), time.item(), device=self.device)
            vector = self.model(t=time, x=x, context=context, mask=mask).to(self.device)
            x += delta_t * vector
            paths.append(x.clone())
        
        paths = torch.stack(paths)

        return paths
    

class TauLeapingSolver:
    def __init__(self, model, config):
        self.model = model # rate model
        self.device = config.device
        self.dim_discrete = config.dim_discrete
        self.vocab_size = config.vocab_size 

    def simulate(self, 
                 time_steps, 
                 source_discrete, 
                 context=None, 
                 mask=None, 
                 max_rate_last_step=False):
        
        k = source_discrete.to(self.device)
        time_steps = time_steps.to(self.device)
        context = context.to(self.device) if context is not None else None
        mask = mask.to(self.device) if mask is not None else None

        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        jumps = [k.clone()]

        for time in time_steps[1:]:
            time = torch.full((k.size(0), 1), time.item(), device=self.device)
            rates = self.model(t=time, k=k, context=context, output_rates=True).to(self.device)
            max_rate = torch.max(rates, dim=2)[1]
            all_jumps = torch.poisson(rates * delta_t).to(self.device) 
            mask =  torch.sum(all_jumps, dim=-1).type_as(k) <= 1
            diff = torch.arange(self.vocab_size, device=self.device).view(1, 1, self.vocab_size) - k[:,:, None]
            net_jumps = torch.sum(all_jumps * diff, dim=-1).type_as(k)
            k += net_jumps * mask
            k = torch.clamp(k, min=0, max=self.vocab_size-1)            
            jumps.append(k.clone())

        jumps = torch.stack(jumps)
        if max_rate_last_step:
            jumps[-1] = max_rate # replace last jump with max rates

        return jumps


class EulerLeapingSolver:
    ''' Euler-Leaping solver combining Euler and Tau-Leaping methods for mixed data
    '''
    def __init__(self, model, config):
        self.model = model # velocity and rate model
        self.device = config.device
        self.dim_discrete = config.dim_discrete
        self.vocab_size = config.vocab_size 

    def simulate(self, 
                 time_steps, 
                 source_continuous, 
                 source_discrete, 
                 context=None, 
                 mask=None, 
                 max_rate_last_step=False):
        
        x = source_continuous.to(self.device)
        k = source_discrete.to(self.device)
        time_steps = time_steps.to(self.device)
        context = context.to(self.device) if context is not None else None
        mask = mask.to(self.device) if mask is not None else None

        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        paths, jumps = [x.clone()], [k.clone()]

        for time in time_steps[1:]:

            time = torch.full((x.size(0), 1), time.item(), device=self.device)

            #...compute velocity and rates:
            vector, rates = self.model(t=time, x=x, k=k, context=context, output_rates=True)
            vector = vector.to(self.device)
            rates = rates.to(self.device)

            #...tau-leaping step:
            max_rate = torch.max(rates, dim=2)[1]
            all_jumps = torch.poisson(rates * delta_t).to(self.device) 
            mask =  torch.sum(all_jumps, dim=-1).type_as(k) <= 1
            diff = torch.arange(self.vocab_size, device=self.device).view(1, 1, self.vocab_size) - k[:,:, None]
            net_jumps = torch.sum(all_jumps * diff, dim=-1).type_as(k)
            k += net_jumps * mask
            k = torch.clamp(k, min=0, max=self.vocab_size-1)            
            jumps.append(k.clone())

            #...euler step:
            x += delta_t * vector
            paths.append(x.clone())
        
        paths = torch.stack(paths)
        jumps = torch.stack(jumps)

        if max_rate_last_step:
            jumps[-1] = max_rate # replace last jump with max rates

        return paths, jumps
    

    

