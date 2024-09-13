import torch 
from dataclasses import dataclass
from cmb.dynamics.utils import right_shape, right_time_size
from torch.nn import CrossEntropyLoss 
from torch.distributions import Categorical 

class ConditionalJumpBridge:
    ''' Conditional Jump Bridge base class
    '''
    def __init__(self, config: dataclass):
        self.config = config
        self.vocab_size = config.vocab_size
        self.loss_fn = CrossEntropyLoss(reduction='mean')

    def sample_coupling(self, batch):
        """ conditional variable z = (k_0, k1) ~ pi(k_0, k_1)
        """		
        self.k0 = batch.source_discrete
        self.k1 = batch.target_discrete
        self.context = None
        self.mask = None

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
        transition_probs = self.telegram_bridge_probability(k , self.k1, self.k0, self.t.squeeze())
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
        return loss     

    def reshape_time(self, t, tensor):
        if isinstance(t, (float, int)): 
            return t
        else: 
            return t.reshape(-1, *([1] * (tensor.dim() - 1)))

    #====================================================================
    # DISCRETE BRIDGE FUNCTIONS
    #====================================================================
    def multivariate_telegram_conditional(self, k, k0, t, t0):
        """
        \begin{equation}
        P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
        \end{equation}

        \begin{equation}
        w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
        \end{equation}

        """
        t = right_time_size(t, k).to(k0.device)
        t0 = right_time_size(t0, k).to(k0.device)
        beta_t = (t - t0) * self.config.gamma
        w_t = torch.exp(-self.vocab_size * beta_t)
        k, k0 = right_shape(k), right_shape(k0)

        kronecker = (k == k0).float()
        probability = 1. / self.vocab_size + w_t[:, None, None] * ((-1. / self.vocab_size) + kronecker)
        return probability

    def telegram_bridge_probability(self, k, k1, k0, t):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        p_k_to_k1 = self.multivariate_telegram_conditional(k1, k, t=1., t0=t)
        p_k0_to_k = self.multivariate_telegram_conditional(k, k0, t=t, t0=0.)
        p_k0_to_k1 = self.multivariate_telegram_conditional(k1, k0, t=1., t0=0.)
        return (p_k_to_k1 * p_k0_to_k) / p_k0_to_k1

