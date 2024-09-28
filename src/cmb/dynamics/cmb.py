import torch 
from dataclasses import dataclass
from torch.nn import MSELoss, CrossEntropyLoss

from cmb.dynamics.utils import OTPlanSampler
from cmb.configs.registered import processes, solvers

class ConditionalMarkovBridge : 
    ''' Conditional Markov Bridge base class for hybrid data
    '''
    def __init__(self, config: dataclass):

        self.config = config
        self.vocab_size = config.data.vocab.size.features

        #...define dynamical processes:
        self.process_continuous = processes['continuous'].get(config.dynamics.continuous.process, None)(config)
        self.process_discrete = processes['discrete'].get(config.dynamics.discrete.process, None)(config)

        #...define solvers:
        solver = solvers.get(config.pipeline.method)
        self.solver = solver(config=config, 
                             dynamics_continuous=self.process_continuous,
                             dynamics_discrete=self.process_discrete)

        #...define losses:
        self.loss_continuous_fn = MSELoss(reduction='sum')
        self.loss_discrete_fn = CrossEntropyLoss(reduction='none')
        self.weight = config.dynamics.loss_weight

        #...logging:
        print('INFO: Conditional Markov Bridge initialized...')
        print('      - continuous process: ', config.dynamics.continuous.process)
        print('      - discrete process: ', config.dynamics.discrete.process)
        print('      - solver method: ', config.pipeline.method)

    def sample_time(self):
        """ sample time: t ~ U[0,1]
        """
        t = torch.rand(self.x1.shape[0], device=self.x1.device).type_as(self.x1)
        self.t = self.reshape_time(t, self.x1)  # shape: (b, 1,...) with len as len(x1)

    def sample_coupling(self, batch):
        """ sample boundary data z = (x_0, x1) ~ pi(x_0, x_1)
        """		
        self.x0 = batch.source_continuous
        self.x1 = batch.target_continuous
        self.k0 = batch.source_discrete
        self.k1 = batch.target_discrete
        self.context_continuous = batch.target_context_continuous if hasattr(batch, 'target_context_continuous') else None
        self.context_discrete = batch.target_context_discrete if hasattr(batch, 'target_context_discrete') else None
        self.mask = batch.target_mask if hasattr(batch, 'target_mask') else torch.ones_like(self.x0[..., 0]).unsqueeze(-1)

    def sample_bridges(self):
        ''' sample paths and jumps from bridges
        '''
        self.xt = self.process_continuous.sample(t=self.t, x0=self.x0, x1=self.x1) if self.process_continuous else None
        self.kt = self.process_discrete.sample(t=self.t, k0=self.k0, k1=self.k1) if self.process_discrete else None 

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

        if self.process_continuous:
            ut = self.process_continuous.drift(t=self.t, 
                                                x=self.xt, 
                                                x0=self.x0, 
                                                x1=self.x1).to(vector.device)
            self.mask = self.mask.to(vector.device)
            ut *= self.mask
            loss += self.loss_continuous_fn(vector, ut) / self.mask.sum()

        if self.process_discrete:
            logits = logits.reshape(-1, self.vocab_size)
            targets = self.k1.reshape(-1).long() 
            targets = targets.to(logits.device)
            self.mask = self.mask.reshape(-1)
            loss_ce = self.loss_discrete_fn(logits, targets).to(logits.device) * self.mask
            loss += self.weight * loss_ce.sum() / self.mask.sum()

        return loss

    def reshape_time(self, t, x):
        if isinstance(t, (float, int)): return t
        else: return t.reshape(-1, *([1] * (x.dim() - 1)))

class BatchOTCMB(ConditionalMarkovBridge):
    def sample_coupling(self, batch):
        OT = OTPlanSampler()	
        self.x0 = batch.source_continuous
        self.x1 = batch.target_continuous
        self.k0 = batch.source_discrete
        self.k1 = batch.target_discrete
        pi = OT.get_map(self.x0, self.x1)
        idx_0, idx_1 = OT.sample_map(pi, self.x0.shape[0], replace=False)
        self.x0, self.x1 = self.x0[idx_0], self.x1[idx_1]
        self.k0, self.k1 = self.k0[idx_0], self.k1[idx_1]
        self.context_continuous = batch.target_context_continuous if hasattr(batch, 'target_context_continuous') else None
        self.context_discrete = batch.target_context_discrete if hasattr(batch, 'target_context_discrete') else None
        self.mask = batch.target_mask if hasattr(batch, 'target_mask') else torch.ones_like(self.x0[..., 0]).unsqueeze(-1)


class BatchSBCMB(ConditionalMarkovBridge):
    def sample_coupling(self, batch):
        regulator = 2 * self.config.dynamics.continuous.sigma**2
        SB = OTPlanSampler(reg=regulator)	
        self.x0 = batch.source_continuous
        self.x1 = batch.target_continuous
        self.k0 = batch.source_discrete
        self.k1 = batch.target_discrete
        pi = SB.get_map(self.x0, self.x1)
        idx_0, idx_1 = SB.sample_map(pi, self.x0.shape[0], replace=False)
        self.x0, self.x1 = self.x0[idx_0], self.x1[idx_1]
        self.k0, self.k1 = self.k0[idx_0], self.k1[idx_1]
        self.context_continuous = batch.target_context_continuous if hasattr(batch, 'target_context_continuous') else None
        self.context_discrete = batch.target_context_discrete if hasattr(batch, 'target_context_discrete') else None
        self.mask = batch.target_mask if hasattr(batch, 'target_mask') else torch.ones_like(self.x0[..., 0]).unsqueeze(-1)

