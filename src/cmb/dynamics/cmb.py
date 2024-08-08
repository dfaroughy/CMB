import torch 
from dataclasses import dataclass
import torch 
from dataclasses import dataclass
from cmb.dynamics.utils import right_shape, right_time_size
from torch.nn import MSELoss, CrossEntropyLoss
from torch.distributions import Categorical 


class ConditionalMarkovBridge :
    ''' Conditional Markov Bridge base class
    '''
    def __init__(self, config: dataclass):
        self.config = config
        self.vocab_size = config.VOCAB_SIZE
        self.alpha = config.ALPHA
        self.loss_continuous_fn = MSELoss(reduction='mean')
        self.loss_discrete_fn = CrossEntropyLoss(reduction='mean')

    def sample_coupling(self, batch):
        """ conditional variable z = (x_0, x1) ~ pi(x_0, x_1)
        """		
        self.x0 = batch.source
        self.x1 = batch.target
        self.s0 = batch.labels 
        self.s1 = batch.context 

    def sample_time(self):
        """ sample time: t ~ U[0,1]
        """
        t = torch.rand(self.x1.shape[0], device=self.x1.device).type_as(self.x1)
        self.t = self.reshape_time(t, self.x1)

    def sample_bridge(self):
        """ sample conditional bridge: x_t ~ p_t(x|x_0, x_1)
        """
        #...continous bridge:
        mean = self.t * self.x1 + (1. - self.t) * self.x0
        std = self.config.SIGMA
        self.continuous_bridge = mean + std * torch.randn_like(mean)
		
        #...discrete bridge:
        s = torch.arange(0, self.vocab_size)
        s = s[None, None, :].repeat((self.s0.size(0), self.s0.size(1), 1)).float()
        s = s.to(self.s0.device)
        transition_probs = self.telegram_bridge_probability(s , self.s1, self.s0, self.t.squeeze())
        self.discrete_bridge = Categorical(transition_probs).sample().to(self.s1.device)

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
        self.sample_bridge()
        self.get_drift()

        vt, logits = model(x=self.continuous_bridge, s=self.discrete_bridge , t=self.t)
        logits = logits.reshape(-1, self.vocab_size)
        targets = self.s1.reshape(-1).long()
        targets = targets.to(logits.device)
        ut = self.drift.to(vt.device)

        loss = self.loss_continuous_fn(vt, ut) + self.alpha * self.loss_discrete_fn(logits, targets)
                                                                                    
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
        beta_t = (t - t0) * self.config.GAMMA
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

