
import torch
from dataclasses import dataclass
from torchdyn.core import NeuralODE
# from cmb.models.utils import EulerSolver, TauLeapingSolver, TransitionRateModel, ContextWrapper

class CFMPipeline:
    
    def __init__(self, 
                 trained_model, 
                 config: dataclass=None,
                 best_epoch_model: bool=False,
                 ):

        self.config = config
        self.trained_model = trained_model
        self.model = self.trained_model.best_epoch_model if best_epoch_model else self.trained_model.last_epoch_model
        self.num_sampling_steps = config.num_timesteps
        self.sampler = config.sampler
        self.device = config.device
        self.has_context = False
        self.time_steps = torch.linspace(0.0, 1.0, config.num_timesteps, device=self.device)

    def generate_samples(self, input, context=None):
        self.source = input.source.to(self.device)
        self.context = input.context.to(self.device) if hasattr(input, 'context') else None
        self.mask = input.mask.to(self.device) if hasattr(input, 'mask') else None
        self.trajectories = self.ODEsolver() 

    @torch.no_grad()
    def ODEsolver(self):

        print('INFO: {} with {} method and steps={}'.format(self.sampler, self.config.solver, self.config.num_timesteps))

        drift = ContextWrapper(self.model, context=self.context, mask=self.mask) 

        if self.sampler == 'EulerSolver':
            node = EulerSolver(vector_field=drift, device=self.device)
        
        trajectories = node.trajectory(x=self.source, t_span=self.time_steps).detach().cpu()

        return trajectories

# class CFMPipeline:
    
#     def __init__(self, 
#                  trained_model, 
#                  config: dataclass=None,
#                  best_epoch_model: bool=False,
#                  ):

#         self.config = config
#         self.trained_model = trained_model
#         self.model = self.trained_model.best_epoch_model if best_epoch_model else self.trained_model.last_epoch_model
#         self.num_sampling_steps = config.num_timesteps
#         self.sampler = config.sampler
#         self.device = config.device
#         self.has_context = False
#         self.time_steps = torch.linspace(0.0, 1.0, config.num_timesteps, device=self.device)

#     def generate_samples(self, input_source):
#         self.source_continuous = input_source.continuous.to(self.device)
#         self.mask = input_source.mask.to(self.device)
#         self.context = input_source.context.to(self.device) if self.has_context else None
#         self.trajectories = self.ODEsolver() 

#     @torch.no_grad()
#     def ODEsolver(self):

#         print('INFO: {} with {} method and steps={}'.format(self.sampler, self.config.solver, self.config.num_timesteps))

#         drift = ContextWrapper(self.model, context=self.context, mask=self.mask)

#         if self.sampler == 'EulerSolver':
#             node = EulerSolver(vector_field=drift, device=self.device)
        
#         trajectories = node.trajectory(x=self.source_continuous, t_span=self.time_steps)
#         x1 = trajectories[-1].detach().cpu()
#         mask = self.mask.unsqueeze(-1).detach().cpu()
#         x1 = torch.cat([x1, mask], dim=-1)

#         return ParticleClouds(dataset=x1, min_num_particles=self.config.min_num_particles, max_num_particles=self.config.max_num_particles, discrete_features=False)
    

class ContextWrapper(torch.nn.Module):
    """ Wraps model to torchdyn compatible format.
    """
    def __init__(self, net, context=None, mask=None):
        super().__init__()
        self.nn = net
        self.context = context
        self.mask = mask
    def forward(self, t, x, k):
        if x is not None: 
            t = t.repeat(x.shape[0])
            t = self.reshape_time_like(t, x)
        else: 
            t = t.repeat(k.shape[0])
            t = self.reshape_time_like(t, k)
        return self.nn(t=t, x=x, k=k, context=self.context, mask=self.mask)

    def reshape_time_like(self, t, tensor):
        if isinstance(t, (float, int)): return t
        else: return t.reshape(-1, *([1] * (tensor.dim() - 1)))


class EulerSolver:
    def __init__(self, vector_field, device):
        self.vector_field = vector_field
        self.device = device

    def trajectory(self, t_span, x, k=None):
        time_steps = len(t_span)
        dt = (t_span[-1] - t_span[0]) / (time_steps - 1)
        trajectory = [x]

        for i in range(1, time_steps):
            t = t_span[i-1]
            x = x + dt * self.vector_field(t, x=x, k=k).to(self.device)
            trajectory.append(x)

        return torch.stack(trajectory)

class CFMPipeline:
    
    def __init__(self, 
                 trained_model, 
                 config: dataclass=None,
                 best_epoch_model: bool=False,
                 ):

        self.config = config
        self.trained_model = trained_model
        self.model = self.trained_model.best_epoch_model if best_epoch_model else self.trained_model.last_epoch_model
        self.num_sampling_steps = config.num_timesteps
        self.sampler = config.sampler
        self.device = config.device
        self.has_context = False
        self.time_steps = torch.linspace(0.0, 1.0, config.num_timesteps, device=self.device)

    def generate_samples(self, input_source, context=None):
        self.source = input_source.to(self.device)
        self.context = context.to(self.device) if self.has_context else None
        self.trajectories = self.ODEsolver() 

    @torch.no_grad()
    def ODEsolver(self):

        print('INFO: {} with {} method and steps={}'.format(self.sampler, self.config.solver, self.config.num_timesteps))

        drift = ContextWrapper(self.model, context=self.context if self.context is not None else None)

        if self.sampler == 'EulerSolver':
            node = EulerSolver(vector_field=drift, device=self.device)
        
        trajectories = node.trajectory(x=self.source, t_span=self.time_steps).detach().cpu()

        return trajectories
    

class CJBPipeline:
    def __init__(self, 
                 trained_model, 
                 config: dataclass=None,
                 best_epoch_model: bool=True
                 ):

        self.config = config
        self.model = trained_model.best_epoch_model if best_epoch_model else trained_model.last_epoch_model
        self.num_sampling_steps = config.num_timesteps
        self.device = config.device
        self.vocab_size = config.vocab_size
        self.time_steps = torch.linspace(0.0, 1.0 - config.time_eps, self.num_sampling_steps, device=self.device)

    @torch.no_grad()
    def generate_samples(self, input_source, context=None):
        self.source = input_source.to(self.device) 
        self.context = context.to(self.device) if context is not None else None

        if self.config.sampler == 'TauLeaping':
            solver = TauLeapingSolver(rate_model=self.model, config=self.config)

        elif self.config.sampler == 'Gillespie':
            solver = GillespieSolver(rate_model=self.model, config=self.config)
        
        jumps, k1 = solver.simulate(time_steps=self.time_steps, k=self.source, context=self.context)
        self.jumps = jumps.detach().cpu()
        self.k1 = k1.detach().cpu()
        
class TauLeapingSolver:
    def __init__(self, rate_model, config):
        self.rate_model = rate_model
        self.device = config.device
        self.dim = config.dim_discrete
        self.vocab_size = config.vocab_size 

    def simulate(self, time_steps, k, context=None):
        delta_t = (time_steps[-1] - time_steps[0]) / (len(time_steps) - 1)
        trajectory = [k]

        for time in time_steps[1:]:
    
            current_state = k.clone()

            # time = torch.full_like(k[:, 0], time)
            time = torch.full((k.size(0), 1), time.item(), device=self.device)
            
            
            rates = self.rate_model(t=time, k=current_state, context=context, output_rates=True).to(self.device)
            max_rate = torch.max(rates, dim=2)[1]

            jumps = torch.poisson(rates * delta_t).to(self.device) 
            mask =  torch.sum(jumps, dim=-1).type_as(current_state) <= 1
            diff = torch.arange(self.vocab_size, device=self.device).view(1, 1, self.vocab_size) - k[:,:, None]
            net_jumps = torch.sum(jumps * diff, dim=-1).type_as(current_state)
            
            k = current_state + net_jumps * mask
            k = torch.clamp(k, min=0, max=self.vocab_size-1)            
            trajectory.append(k.clone())

        return torch.stack(trajectory), max_rate
    
class GillespieSolver:
    pass    