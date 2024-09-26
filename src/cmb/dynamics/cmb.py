import torch 
from dataclasses import dataclass
from torch.nn import MSELoss, CrossEntropyLoss
from torch.distributions import Categorical 

from cmb.dynamics.utils import OTPlanSampler
from cmb.dynamics.processes import TelegraphBridge, LinearBridge, SchrodingerBridge

class ConditionalMarkovBridge :
    ''' Conditional Markov Bridge base class for hybrid data
    '''
    def __init__(self, config: dataclass):

        self.config = config.dynamics
        self.vocab_size = config.data.vocab_size.features
        
        #... dynamics:
        self.ref_bridge_continuous = LinearBridge(config)
        self.ref_bridge_discrete = TelegraphBridge(config)

        #...losses:
        self.loss_continuous_fn = MSELoss(reduction='sum')
        self.loss_discrete_fn = CrossEntropyLoss(reduction='sum')

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

    def sample_bridge(self):
        self.xt = self.ref_bridge_continuous.sample(t=self.t, x0=self.x0, x1=self.x1)
        self.kt = self.ref_bridge_discrete.sample(t=self.t, k0=self.k0, k1=self.k1)

    def loss(self, model, batch):
        """ conditional flow-mathcing MSE loss + jump-matching CE loss
        """
        self.sample_coupling(batch)
        self.sample_time() 
        self.sample_bridge()

        vt, logits = model(t=self.t, 
                           x=self.xt, 
                           k=self.kt, 
                           context_continuous=self.context_continuous, 
                           context_discrete=self.context_discrete, 
                           mask=self.mask)
        
        logits = logits.reshape(-1, self.vocab_size) 
        targets = self.k1.reshape(-1).long() 
        targets = targets.to(logits.device)

        ut = self.ref_bridge_continuous.drift(t=self.t, 
                                              x=self.xt, 
                                              x0=self.x0, 
                                              x1=self.x1)
        ut = ut * self.mask
        ut = ut.to(vt.device)
        loss_1 = self.loss_continuous_fn(vt, ut) 
        loss_2 = self.loss_discrete_fn(logits, targets)
        loss = loss_1 + self.config.lam * loss_2
        return loss / self.mask.sum()

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
    def __init__(self, config: dataclass):
        super().__init__(config)
        self.ref_bridge_continuous = SchrodingerBridge(config)
        self.ref_bridge_discrete = TelegraphBridge(config)

    def sample_coupling(self, batch):
        regulator = 2 * self.config.sigma**2
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

