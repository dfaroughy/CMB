
import torch
from dataclasses import dataclass
from torchdyn.core import NeuralODE
from cmb.models.utils import EulerSolver, ContextWrapper

class CFMPipeline:
    
    def __init__(self, 
                 trained_model, 
                 config: dataclass=None,
                 best_epoch_model: bool=False,
                 ):

        self.config = config
        self.trained_model = trained_model
        self.model = self.trained_model.best_epoch_model if best_epoch_model else self.trained_model.last_epoch_model
        self.num_sampling_steps = config.NUM_SAMPLING_STEPS
        self.sampler = config.SAMPLER
        self.device = config.DEVICE
        self.has_context = True if config.DIM_CONTEXT > 0 else False
        self.time_steps = torch.linspace(0.0, 1.0, config.NUM_SAMPLING_STEPS, device=self.device)

    def generate_samples(self, input_source, context=None):
        self.source = input_source.to(self.device)
        self.context = context.to(self.device) if self.has_context else None
        self.trajectories = self.ODEsolver() 

    @torch.no_grad()
    def ODEsolver(self):

        print('INFO: {} with {} method and steps={}'.format(self.sampler, self.config.SOLVER, self.config.NUM_SAMPLING_STEPS))

        drift = ContextWrapper(self.model, context=self.context if self.context is not None else None)

        if self.sampler == 'EulerSolver':
            node = EulerSolver(vector_field=drift, device=self.device)
        
        elif self.sampler == 'NeuralODE':
            node = NeuralODE(vector_field=drift, 
                             solver=self.config.SOLVER, 
                             sensitivity=self.config.SENSITIVITY, 
                             seminorm=True if self.config.SOLVER=='dopri5' else False,
                             atol=self.config.ATOL if self.solver=='dopri5' else None, 
                             rtol=self.config.RTOL if self.solver=='dopri5' else None)        
        else:
            raise ValueError('Invalid sampler method.')
        
        trajectories = node.trajectory(x=self.source, t_span=self.time_steps).detach().cpu()

        return trajectories
    

class CJBPipeline:
    
    def __init__(self, 
                 trained_model, 
                 config: dataclass=None,
                 best_epoch_model: bool=True,
                 ):

        self.config = config
        self.model = trained_model.best_epoch_model if best_epoch_model else trained_model.last_epoch_model
        self.num_sampling_steps = config.NUM_TIMESTEPS
        self.sampler = config.SAMPLER
        self.device = config.DEVICE
        self.vocab_size = config.VOCAB_SIZE
        self.has_context = True if config.DIM_CONTEXT > 0 else False
        self.time_steps = torch.linspace(0.0, 1.0 - config.TIME_EPS, self.num_sampling_steps, device=self.device)

    @torch.no_grad()
    def generate_samples(self, input_source, context=None):
        self.source = input_source.to(self.device) 
        self.context = context.to(self.device) if self.has_context else None
        self.jumps = self.MarkovSolver() 

    @torch.no_grad()
    def MarkovSolver(self):
        rate = TransitionRateModel(self.model, self.config)
        rate = ContextWrapper(rate, context=self.context if self.context is not None else None)
        tauleap = TauLeapingSolver(transition_rate=rate, device=self.device)        
        return tauleap.simulate(x=self.source, t_span=self.time_steps).detach().cpu()