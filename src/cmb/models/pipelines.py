
import torch
from torchdyn.core import NeuralODE
from tqdm.auto import tqdm
from dataclasses import dataclass

from cmb.models.utils import TorchdynWrapper

class CFMPipeline:
    
    def __init__(self, 
                 trained_model, 
                 preprocessor: object=None,
                 postprocessor: object=None,
                 config: dataclass=None,
                 solver: str=None,
                 num_sampling_steps: int=None,
                 sensitivity: str=None,
                 atol: float=None,
                 rtol: float=None,
                 reverse_time_flow: bool=False,
                 best_epoch_model: bool=False,
                 batch_size: int=None
                 ):
        
        self.trained_model = trained_model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model = self.trained_model.best_epoch_model if best_epoch_model else self.trained_model.last_epoch_model

        self.t0 = config.T1 if reverse_time_flow else config.T0
        self.t1 = config.T0 if reverse_time_flow else config.T1
        self.solver = config.SOLVER if solver is None else solver
        self.num_sampling_steps = config.NUM_SAMPLING_STEPS if num_sampling_steps is None else num_sampling_steps
        self.sensitivity = config.SENSITIVITY if sensitivity is None else sensitivity
        self.atol = config.ATOL if atol is None else atol
        self.rtol = config.RTOL if rtol is None else rtol
        self.device = config.DEVICE
        self.time_steps = torch.linspace(self.t0, self.t1, self.num_sampling_steps, device=self.device)
        self.batch_size = config.BATCH_SIZE if batch_size is None else batch_size  

    @torch.no_grad()
    def generate_samples(self, input_source, context=None):
        self.source = self._preprocess(input_source)
        self.context = context.to(self.device) if context is not None else None
        self.trajectories = self._postprocess(self._ODEsolver())  

    def _preprocess(self, samples):
        samples = samples.to(self.device)
        if self.preprocessor is not None:
            self.stats = self.trained_model.dataloader.datasets.summary_stats
            samples = self.preprocessor(samples, methods=self.trained_model.dataloader.datasets.preprocess_methods, summary_stats=self.stats)
            samples.preprocess()
            return samples.features
        else:
            return samples

    def _postprocess(self, samples):
        if self.postprocessor is not None:
            self.stats = self.trained_model.dataloader.datasets.summary_stats
            self.postprocess_methods = ['inverse_' + method for method in self.trained_model.dataloader.datasets.preprocess_methods[::-1]]
            samples = self.postprocessor(samples, methods=self.postprocess_methods, summary_stats=self.stats)
            samples.postprocess()
            return samples.features
        else:
            return samples

    @torch.no_grad()
    def _ODEsolver(self):
        print('INFO: neural ODE solver with {} method and steps={}'.format(self.solver, self.num_sampling_steps))

        if self.solver == 'dopri5':
            assert self.atol is not None and self.rtol is not None, 'atol and rtol must be specified for the chosen solver'

        drift = TorchdynWrapper(self.model, context=self.context if self.context is not None else None)

        node = NeuralODE(vector_field=drift, 
                        solver=self.solver, 
                        sensitivity=self.sensitivity, 
                        seminorm=True if self.solver=='dopri5' else False,
                        atol=self.atol if self.solver=='dopri5' else None, 
                        rtol=self.rtol if self.solver=='dopri5' else None)
        
        trajectories = node.trajectory(x=self.source, t_span=self.time_steps).detach().cpu()

        return trajectories
