import torch 
from dataclasses import dataclass
from cmb.dynamics.utils import right_shape, right_time_size
from torch.nn import CrossEntropyLoss 
from torch.distributions import Categorical 

from cmb.dynamics.processes import TelegraphProcess

class ConditionalJumpBridge:
    ''' Conditional Jump Bridge base class
    '''
    def __init__(self, config: dataclass):
        self.config = config
        self.vocab_size = config.vocab_size
        self.ref_process = TelegraphProcess(config)
        self.loss_fn = CrossEntropyLoss(reduction='sum')

    def sample_coupling(self, batch):
        """ conditional variable z = (k_0, k1) ~ pi(k_0, k_1)
        """		
        self.k0 = batch.source_discrete
        self.k1 = batch.target_discrete
        self.x0 = batch.source_continuous
        self.x1 = batch.target_continuous
        self.context_continuous = batch.target_context_continuous if hasattr(batch, 'target_context_continuous') else None
        self.context_discrete = batch.target_context_discrete if hasattr(batch, 'target_context_discrete') else None
        self.mask = batch.target_mask if hasattr(batch, 'target_mask') else torch.ones_like(self.x0[..., 0]).unsqueeze(-1)

    def sample_time(self):
        """ sample time: t ~ U[0,1]
        """
        t = torch.rand(self.k1.shape[0], device=self.k1.device).type_as(self.k1)
        self.t = self.reshape_time(t, self.k1)

    def sample_bridge(self):
        k = torch.arange(0, self.vocab_size)

        if self.k0.dim() == 1:
            self.k0 = self.k0.unsqueeze(1)  
            self.k1 = self.k1.unsqueeze(1)

        k = k[None, None, :].repeat((self.k0.size(0), self.k0.size(1), 1)).float()
        k = k.to(self.k0.device)
        transition_probs = self.ref_process.bridge_probability(k , self.k1, self.k0, self.t.squeeze())
        self.bridge = Categorical(transition_probs).sample().to(self.k1.device)

    def loss(self, model, batch):
        self.sample_coupling(batch)
        self.sample_time() 
        self.sample_bridge()
        logits = model(t=self.t, k=self.bridge, context=self.context, mask=self.mask)
        logits = logits.reshape(-1, self.vocab_size)
        targets = self.k1.reshape(-1).long()
        targets = targets.to(logits.device)
        loss = self.loss_fn(logits, targets)
        return loss / self.mask.sum()

    def reshape_time(self, t, tensor):
        if isinstance(t, (float, int)): 
            return t
        else: 
            return t.reshape(-1, *([1] * (tensor.dim() - 1)))
