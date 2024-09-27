import torch 
from dataclasses import dataclass
from torch.nn.functional import softmax
from torch.distributions import Categorical 

# from cmb.dynamics.utils import right_shape, right_time_size

class LinearBridge:
    ''' Linear bridge for continuous states. 
        Equivalent to vanilla Flow-matching
    '''
    def __init__(self, config: dataclass):
        self.config = config.dynamics
        self.time_epsilon = config.pipeline.time_eps

    def sample(self, t, x0, x1):
        x = t * x1 + (1. - t) * x0
        std = self.config.sigma 
        return x + std * torch.randn_like(x)

    def drift(self, t, x, x0, x1):
        A = 0.0
        B = 1.0
        C = -1.0 
        return A * x + B * x1 + C * x0

    def diffusion(self, t):
        return self.config.sigma
    
class SchrodingerBridge:
    ''' Schrodinger bridge for continuous states
    '''
    def __init__(self, config: dataclass):
        self.config = config.dynamics
        self.time_epsilon = config.pipeline.time_eps

    def sample(self, t, x0, x1):
        x = t * x1 + (1. - t) * x0
        std = self.config.sigma * torch.sqrt(t * (1. - t))
        return x + std * torch.randn_like(x)

    def drift(self, t, x, x0, x1):
        A = (1 - 2 * t) / ( t * (1 - t))
        B = t**2 / ( t * (1 - t))
        C = -1 * (1 - t)**2 / ( t * (1 - t))
        return A * x + B * x1 + C * x0
    
    def diffusion(self, t):
        return self.config.sigma * torch.sqrt(t * (1. - t))


class TelegraphBridge:
    ''' Multivariate Telegraph Bridge for discrete states
    '''
    def __init__(self, config: dataclass):
        self.config = config.dynamics
        self.vocab_size = config.data.vocab_size.features
        self.time_epsilon = config.pipeline.time_eps
        self.ignore_idx = config.data.vocab_size.mask_idx
    
    def sample(self, t, k0, k1):
        transition_probs = self.probability(t, k0, k1)
        return Categorical(transition_probs).sample().to(k1.device)
    
    def rate(self, t, k, logits, ignore_idx=0):

        logits[..., ignore_idx] = float('-inf')               # ignore zero-padding
    
        qx = softmax(logits, dim=2)                           # softmax to get the transition probabilities for all states
        qy = torch.gather(qx, 2, k.long().unsqueeze(-1))       # get probabilities for the current state `k`

        #...apply the Telegraph rates:

        S = self.vocab_size - 1                               # -1 ignore zero-padding
        t = t.squeeze()
        t1 = 1. - self.time_epsilon
        wt = torch.exp(-S * self.config.gamma * (t1 - t) )
        A = 1.0
        B = (wt * S) / (1. - wt)
        C = wt

        rate = A + B[:, None, None] * qx + C[:, None, None] * qy
        mask = (k != ignore_idx).float().unsqueeze(-1)                  # Mask for ignoring transitions from ignored states
        rate = rate * mask                                              # Zero out the rates for transitions involving ignored states

        return rate

    def probability(self, t, k0, k1):
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
        S = self.vocab_size - 1
        t_out = right_time_size(t_out, k_out).to(k_in.device)
        t_in = right_time_size(t_in, k_out).to(k_in.device)
        w_t = torch.exp(- S * self.config.gamma * (t_out - t_in))
        k_out, k_in = right_shape(k_out), right_shape(k_in)
        kronecker = (k_out == k_in).float()
        prob = 1. / S + w_t[:, None, None] * ((-1. / S) + kronecker)
        return prob

right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
right_time_size = lambda t, x: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

def where_to_go_x(x, vocab_size):
    x_to_go = torch.arange(0, vocab_size)
    x_to_go = x_to_go[None, None, :].repeat((x.size(0), x.size(1), 1)).float()
    x_to_go = x_to_go.to(x.device)
    return x_to_go
