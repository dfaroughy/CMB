import torch 
from dataclasses import dataclass
from torch.nn import MSELoss, CrossEntropyLoss
from torch.distributions import Categorical 

from cmb.dynamics.utils import OTPlanSampler
from cmb.dynamics.processes import TelegraphProcess

class ConditionalMarkovBridge :
    ''' Conditional Markov Bridge base class
    '''
    def __init__(self, config: dataclass):

        self.config = config.dynamics
        self.vocab_size = config.data.vocab_size.features

        self.ref_process = TelegraphProcess(config)
        self.loss_continuous_fn = MSELoss(reduction='sum')
        self.loss_discrete_fn = CrossEntropyLoss(reduction='sum')
    
    def sample_coupling(self, batch):
        """ conditional variable z = (x_0, x1) ~ pi(x_0, x_1)
        """		
        self.x0 = batch.source_continuous
        self.x1 = batch.target_continuous
        self.k0 = batch.source_discrete
        self.k1 = batch.target_discrete
        self.context_continuous = batch.target_context_continuous if hasattr(batch, 'target_context_continuous') else None
        self.context_discrete = batch.target_context_discrete if hasattr(batch, 'target_context_discrete') else None
        self.mask = batch.target_mask if hasattr(batch, 'target_mask') else torch.ones_like(self.x0[..., 0]).unsqueeze(-1)

    def sample_time(self):
        """ sample time: t ~ U[0,1]
        """
        t = torch.rand(self.x1.shape[0], device=self.x1.device).type_as(self.x1)
        self.t = self.reshape_time(t, self.x1)

    def sample_bridge(self):
        """ sample continuous features from gaussian probability path: x_t ~ p_t(x|x_0, x_1)
        """
        #...continuous:
        mean = self.t * self.x1 + (1. - self.t) * self.x0
        std = self.config.sigma
        self.bridge_continuous = mean + std * torch.randn_like(mean)
	
        #...discrete:        
        k = torch.arange(0, self.vocab_size)
        if self.k0.dim() == 1: self.k0 = self.k0.unsqueeze(1)  # Add an extra dimension if needed
        if self.k1.dim() == 1: self.k1 = self.k1.unsqueeze(1)

        k = k[None, None, :].repeat(self.k0.size(0), self.k0.size(1), 1).float()
        k = k.to(self.k0.device)


        transition_probs = self.ref_process.bridge_probability(k, self.k1, self.k0, self.t.squeeze())
        self.bridge_discrete = Categorical(transition_probs).sample().to(self.k1.device)

    def get_drift(self):
        """ conditional drift u_t(x|x_0,x_1)
        """
        A = 0.
        B = 1.
        C = -1.
        self.drift = A * self.bridge_continuous + B * self.x1 + C * self.x0

    def loss(self, model, batch):
        """ conditional flow-mathcing MSE loss + jump-matching CE loss
        """
        self.sample_coupling(batch)
        self.sample_time() 
        self.sample_bridge()
        self.get_drift()

        vt, logits = model(t=self.t, 
                           x=self.bridge_continuous, 
                           k=self.bridge_discrete, 
                           context_continuous=self.context_continuous, 
                           context_discrete=self.context_discrete, 
                           mask=self.mask,                    
                           output_rates=False)
        
        logits = logits.reshape(-1, self.vocab_size)
        targets = self.k1.reshape(-1).long()
        targets = targets.to(logits.device)
        ut = self.drift.to(vt.device)
        loss = self.loss_continuous_fn(vt, ut) + self.config.lam * self.loss_discrete_fn(logits, targets)

        return loss / self.mask.sum()

    def reshape_time(self, t, x):
        if isinstance(t, (float, int)): return t
        else: return t.reshape(-1, *([1] * (x.dim() - 1)))

class OTCMB(ConditionalMarkovBridge):
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


class SBCMB(ConditionalMarkovBridge):
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

    def sample_continuous_bridge(self):
        self.mean = self.t * self.x1 + (1 - self.t) * self.x0
        std = self.config.sigma * torch.sqrt(self.t * (1 - self.t))
        self.continuous_bridge = self.mean + std * torch.randn_like(self.mean)
		
    def get_drift(self):
        """ conditional drift u_t(x|x_0,x_1)
        """
        A = (1 - 2 * self.t) / ( self.t * (1 - self.t))
        B = self.t**2 / ( self.t * (1 - self.t))
        C = -1 * (1 - self.t)**2 / ( self.t * (1 - self.t))

        self.drift = A * self.continuous_bridge + B * self.x1 + C * self.x0