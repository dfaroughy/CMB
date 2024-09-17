import torch 
from dataclasses import dataclass
import torch 
from dataclasses import dataclass
from cmb.dynamics.utils import right_shape, right_time_size
from torch.nn import MSELoss, CrossEntropyLoss
from torch.distributions import Categorical 
from cmb.dynamics.utils import OTPlanSampler


class ConditionalMarkovBridge :
    ''' Conditional Markov Bridge base class
    '''
    def __init__(self, config: dataclass):
        self.config = config
        self.vocab_size = config.vocab_size
        self.lam = config.lam                             # weight for discrete loss wrt continuous loss
        self.loss_continuous_fn = MSELoss(reduction='mean')
        self.loss_discrete_fn = CrossEntropyLoss(reduction='mean')

    def sample_coupling(self, batch):
        """ conditional variable z = (x_0, x1) ~ pi(x_0, x_1)
        """		
        self.x0 = batch.source_continuous
        self.x1 = batch.target_continuous
        self.k0 = batch.source_discrete
        self.k1 = batch.target_discrete
        self.context = None
        self.mask = None

    def sample_time(self):
        """ sample time: t ~ U[0,1]
        """
        t = torch.rand(self.x1.shape[0], device=self.x1.device).type_as(self.x1)
        self.t = self.reshape_time(t, self.x1)

    def sample_continuous_bridge(self):
        """ sample conditional bridge: x_t ~ p_t(x|x_0, x_1)
        """
        mean = self.t * self.x1 + (1. - self.t) * self.x0
        std = self.config.sigma
        self.continuous_bridge = mean + std * torch.randn_like(mean)
	
    def sample_discrete_bridge(self):
        """ sample conditional bridge: x_t ~ p_t(x|x_0, x_1)
        """
        
        k = torch.arange(0, self.vocab_size)

        # Ensure k0 has at least 2 dimensions
        if self.k0.dim() == 1:
            self.k0 = self.k0.unsqueeze(1)  # Add an extra dimension if needed
        if self.k1.dim() == 1:
            self.k1 = self.k1.unsqueeze(1)

        # Adjust the repeat based on the actual dimensions of k0 and k1
        k = k[None, None, :].repeat(self.k0.size(0), self.k0.size(1), 1).float()
        k = k.to(self.k0.device)
        transition_probs = self.telegram_bridge_probability(k, self.k1, self.k0, self.t.squeeze())
        self.discrete_bridge = Categorical(transition_probs).sample().to(self.k1.device)

    def get_drift(self):
        """ conditional drift u_t(x|x_0,x_1)
        """
        A = 0.
        B = 1.
        C = -1.
        self.drift = A * self.continuous_bridge + B * self.x1 + C * self.x0

    def loss(self, model, batch):
        """ conditional flow-mathcing MSE loss
        """
        self.sample_coupling(batch)
        self.sample_time() 
        self.sample_continuous_bridge()
        self.sample_discrete_bridge()
        self.get_drift()
        vt, logits = model(t=self.t, x=self.continuous_bridge, k=self.discrete_bridge, context=self.context, mask=self.mask, output_rates=False)
        logits = logits.reshape(-1, self.vocab_size)
        targets = self.k1.reshape(-1).long()
        targets = targets.to(logits.device)
        ut = self.drift.to(vt.device)
        loss = self.loss_continuous_fn(vt, ut) + self.lam * self.loss_discrete_fn(logits, targets)

        return loss

    def reshape_time(self, t, x):
        if isinstance(t, (float, int)): return t
        else: return t.reshape(-1, *([1] * (x.dim() - 1)))

    #====================================================================
    # DISCRETE BRIDGE FUNCTIONS
    #====================================================================

    def multivariate_telegram_conditional(self, x, x0, t, t0):
        """
        \begin{equation}
        P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
        \end{equation}

        \begin{equation}
        w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
        \end{equation}

        """
        t = right_time_size(t,x).to(x0.device)
        t0 = right_time_size(t0,x).to(x0.device)
        beta_t = (t - t0) * self.config.gamma
        w_t = torch.exp(-self.vocab_size * beta_t)
        x, x0 = right_shape(x), right_shape(x0)

        kronecker = (x == x0).float()
        probability = 1. / self.vocab_size + w_t[:, None, None] * ((-1. / self.vocab_size) + kronecker)
        return probability

    def telegram_bridge_probability(self, x, x1, x0, t):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        p_x_to_x1 = self.multivariate_telegram_conditional(x1, x, t=1., t0=t)
        p_x0_to_x = self.multivariate_telegram_conditional(x, x0, t=t, t0=0.)
        p_x0_to_x1 = self.multivariate_telegram_conditional(x1, x0, t=1., t0=0.)
        return (p_x_to_x1 * p_x0_to_x) / p_x0_to_x1


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
        self.context = batch.context if hasattr(batch, 'context') else None
        self.mask = batch.mask if hasattr(batch, 'mask') else None


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
        self.context = batch.context if hasattr(batch, 'context') else None
        self.mask = batch.mask if hasattr(batch, 'mask') else None

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