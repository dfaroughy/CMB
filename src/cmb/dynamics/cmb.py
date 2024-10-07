import torch 
from dataclasses import dataclass
from torch.nn import MSELoss, CrossEntropyLoss

from cmb.dynamics.utils import OTPlanSampler
from cmb.configs.registered_processes import processes
from cmb.configs.registered_solvers import solvers

class ConditionalMarkovBridge : 
    ''' Conditional Markov Bridge base class for hybrid data
    '''
    def __init__(self, config: dataclass):

        self.config = config
        self.vocab_size = config.data.vocab.size.features

        #...get dynamical process from config:
    
        if hasattr(config.dynamics, 'continuous'): 
            self.process_continuous = processes['continuous'].get(config.dynamics.continuous.process)(config)
            self.loss_continuous_fn = MSELoss(reduction='none')
        
        if hasattr(config.dynamics, 'discrete'): 
            self.process_discrete = processes['discrete'].get(config.dynamics.discrete.process)(config)
            self.loss_discrete_fn = CrossEntropyLoss(reduction='none')
            self.weight = config.dynamics.loss_weight if hasattr(config.dynamics, 'loss_weight') else 1.0

        #...get solver from config:
            
        solver = solvers.get(config.pipeline.method)
        self.solver = solver(config=config, 
                             dynamics_continuous=getattr(self, 'process_continuous', None),
                             dynamics_discrete=getattr(self, 'process_discrete', None)
                            )
        #...logging:
        print('INFO: Conditional Markov Bridge initialized...')
        print('      - continuous process: ', config.dynamics.continuous.process if hasattr(self, 'process_continuous') else None)
        print('      - discrete process: ', config.dynamics.discrete.process if hasattr(self, 'process_discrete') else None)
        print('      - solver method: ', config.pipeline.method)

    def sample_time(self):
        """ sample time: t ~ U[0,1]
        """
        t = torch.rand(self.x1.shape[0], device=self.x1.device).type_as(self.x1)
        self.t = self.reshape_time(t, self.x1)  # shape: (b, 1,...) with len as len(x1)

    def sample_coupling(self, batch):
        """ sample boundary data z = (x_0, x1) ~ pi(x_0, x_1)
        """		
        #...continuous coupling:
        self.x0 = getattr(batch, 'source_continuous', None)
        self.x1 = getattr(batch, 'target_continuous', None)

        #...discrete coupling:
        self.k0 = getattr(batch, 'source_discrete', None)
        self.k1 = getattr(batch, 'target_discrete', None)

        #...mask and context:
        self.mask = getattr(batch, 'target_mask', torch.ones_like(self.x0[..., 0]).unsqueeze(-1))
        self.context_continuous = getattr(batch, 'target_context_continuous', None)
        self.context_discrete = getattr(batch, 'target_context_discrete', None)

    def sample_bridges(self):
        ''' sample paths and jumps from bridges
        '''
        self.xt = self.process_continuous.sample(t=self.t, x0=self.x0, x1=self.x1) if hasattr(self, 'process_continuous') else None
        self.kt = self.process_discrete.sample(t=self.t, k0=self.k0, k1=self.k1) if hasattr(self, 'process_discrete') else None 

    def loss_weight(self, device):
        ''' time-dependent relative weight function in front the discrete CE loss
        '''
        if self.config.dynamics.loss_weight_fn =='linear':  
            weight = self.weight * (1.0 - self.t)
            weight = weight.to(device)
        elif self.config.dynamics.loss_weight_fn =='parabolic':  
            weight = self.weight * 4 * self.t * (1.0 - self.t)
            weight = weight.to(device)
        else: 
            weight = self.weight
        return weight
    
    def loss(self, model, batch):
        
        loss = 0.0

        self.sample_coupling(batch)
        self.sample_time() 
        self.sample_bridges()

        vector, logits = model(t=self.t, 
                               x=self.xt, 
                               k=self.kt, 
                               context_continuous=self.context_continuous, 
                               context_discrete=self.context_discrete, 
                               mask=self.mask)

        self.mask = self.mask.to(vector.device)
        
        if hasattr(self, 'process_continuous'):
            ut = self.process_continuous.drift(t=self.t, 
                                               x=self.xt, 
                                               x0=self.x0, 
                                               x1=self.x1).to(vector.device)
            
            loss_mse = self.loss_continuous_fn(vector, ut) * self.mask
            loss +=  loss_mse.sum() / self.mask.sum()

        if hasattr(self, 'process_discrete'):
            logits = logits.reshape(-1, self.vocab_size)
            targets = self.k1.reshape(-1).long() 
            targets = targets.to(logits.device)
            self.mask = self.mask.reshape(-1)
            lambda_t = self.loss_weight(device=logits.device)

            loss_ce = lambda_t * self.loss_discrete_fn(logits, targets)
            loss_ce = loss_ce * self.mask
            loss += loss_ce.sum() / self.mask.sum()

        return loss

    def reshape_time(self, t, x):
        if isinstance(t, (float, int)): return t
        else: return t.reshape(-1, *([1] * (x.dim() - 1)))

class BatchOTCMB(ConditionalMarkovBridge):
    def sample_coupling(self, batch):
        OT = OTPlanSampler()	
        self.x0 = batch.source_continuous
        self.x1 = batch.target_continuous
        self.k0 = getattr(batch, 'source_discrete', None)
        self.k1 = getattr(batch, 'target_discrete', None)
        self.context_continuous = getattr(batch, 'target_context_continuous', None)
        self.context_discrete = getattr(batch, 'target_context_discrete', None)
        self.mask = getattr(batch, 'target_mask', torch.ones_like(self.x0[..., 0]).unsqueeze(-1))

        #...OT pairing
        pi = OT.get_map(self.x0, self.x1)
        idx_0, idx_1 = OT.sample_map(pi, self.x0.shape[0], replace=False)
        self.x0, self.x1 = self.x0[idx_0], self.x1[idx_1]
        self.k0, self.k1 = self.k0[idx_0], self.k1[idx_1]


class BatchEntropicOTCMB(ConditionalMarkovBridge):
    def sample_coupling(self, batch):
        regulator = 2 * self.config.dynamics.continuous.sigma**2
        EOT = OTPlanSampler(reg=regulator)	
        self.x0 = batch.source_continuous
        self.x1 = batch.target_continuous
        self.k0 = getattr(batch, 'source_discrete', None)
        self.k1 = getattr(batch, 'target_discrete', None)
        self.context_continuous = getattr(batch, 'target_context_continuous', None)
        self.context_discrete = getattr(batch, 'target_context_discrete', None)
        self.mask = getattr(batch, 'target_mask', torch.ones_like(self.x0[..., 0]).unsqueeze(-1))

        #...OT pairing
        pi = EOT.get_map(self.x0, self.x1)
        idx_0, idx_1 = EOT.sample_map(pi, self.x0.shape[0], replace=False)
        self.x0, self.x1 = self.x0[idx_0], self.x1[idx_1]
        self.k0, self.k1 = self.k0[idx_0], self.k1[idx_1]

