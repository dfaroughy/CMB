import torch 
from dataclasses import dataclass
from torch.nn.functional import softmax
from torch.distributions import Categorical 

class FlowMatching:
    ''' Conditional Flow-Matching for continuous states. 
    '''
    def __init__(self, config: dataclass):
        self.config = config.dynamics.continuous
        self.time_epsilon = config.pipeline.time_eps

    def sample(self, t, x0, x1):
        x = t * x1 + (1. - t) * x0
        z = torch.randn_like(x)
        std = self.config.sigma 
        return x + std * z

    def drift(self, t, x, x0, x1):
        A = 0.0
        B = 1.0
        C = -1.0 
        return A * x + B * x1 + C * x0

    def diffusion(self, t):
        return 0.0
    
class SchrodingerBridge:
    ''' Schrodinger bridge for continuous states
    '''
    def __init__(self, config: dataclass):
        self.config = config.dynamics.continuous
        self.time_epsilon = config.pipeline.time_eps

    def sample(self, t, x0, x1):
        x = t * x1 + (1. - t) * x0
        z = torch.randn_like(x)
        std = self.config.sigma * torch.sqrt(t * (1. - t))
        return x + std * z

    def drift(self, t, x, x0, x1):
        A = (1 - 2 * t) / ( t * (1 - t))
        B = t**2 / ( t * (1 - t))
        C = -1 * (1 - t)**2 / ( t * (1 - t))
        return A * x + B * x1 + C * x0
    
    def diffusion(self, t):
        return self.config.sigma * torch.sqrt(t * (1. - t))


class TelegraphProcess:
    ''' Multivariate Telegraph Process for discrete states
    '''
    def __init__(self, config: dataclass):
        self.config = config.dynamics.discrete
        self.time_epsilon = config.pipeline.time_eps
        self.vocab_size = config.data.vocab.size.features
    
    def sample(self, t, k0, k1):
        transition_probs = self.transition_probability(t, k0, k1)
        return Categorical(transition_probs).sample().to(k1.device)
    
    def rate(self, t, k, logits):
        ''' t: (b, 1) time tensor
            k: (b, n, 1) current state tensor
            logits: (b, n, vocab_size) logits tensor
        '''

        assert (k >= 0).all() and (k < self.vocab_size).all(), "Values in `k` outside of bound!"
        
        qx = softmax(logits, dim=2)             # softmax to get the transition probabilities for all states
        qy = torch.gather(qx, 2, k.long())      # get probabilities for the current state `k`

        #...Telegraph process rates:

        S = self.vocab_size                             
        t = t.squeeze()
        t1 = 1. - self.time_epsilon
        wt = torch.exp(-S * self.config.gamma * (t1 - t) )
        A = 1.0
        B = (wt * S) / (1. - wt)
        C = wt
        rate = A + B[:, None, None] * qx + C[:, None, None] * qy
        return rate

    def transition_probability(self, t, k0, k1):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        #...reshape input tenors:
        t = t.squeeze()
        if k0.dim() == 1: k0 = k0.unsqueeze(1)  # Add an extra dimension if needed
        if k1.dim() == 1: k1 = k1.unsqueeze(1)

        #...set state configurations:
        k = torch.arange(0, self.vocab_size)  # shape: (vocab_size,)
        k = k[None, None, :].repeat(k0.size(0), k0.size(1), 1).float()
        k = k.to(k0.device)

        #...compute probabilities:
        p_k_to_k1 = self.conditional_probability(t, 1.0, k, k1)
        p_k0_to_k = self.conditional_probability(0.0, t, k0, k)
        p_k0_to_k1 = self.conditional_probability(0.0, 1.0, k0, k1)
        
        return (p_k_to_k1 * p_k0_to_k) / p_k0_to_k1


    def conditional_probability(self, t_in, t_out, k_in, k_out):
        """
        \begin{equation}
        P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
        \end{equation}

        \begin{equation}
        w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
        \end{equation}

        """
        S = self.vocab_size
        t_out = right_time_size(t_out, k_out).to(k_in.device)
        t_in = right_time_size(t_in, k_out).to(k_in.device)
        w_t = torch.exp(- S * self.config.gamma * (t_out - t_in))
        k_out, k_in = right_shape(k_out), right_shape(k_in)
        kronecker = (k_out == k_in).float()
        prob = 1. / S + w_t[:, None, None] * ((-1. / S) + kronecker)
        return prob

right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
right_time_size = lambda t, x: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)
