import torch 
from dataclasses import dataclass
from torch.nn.functional import softmax

from cmb.dynamics.utils import right_shape, right_time_size

class TelegraphProcess:
    ''' Multivariate Telegraph Process
    '''
    def __init__(self, config: dataclass):
        self.config = config.dynamics
        self.vocab_size = config.data.vocab_size.features
        self.time_epsilon = config.pipeline.time_eps

    def conditional_probability(self, k, k0, t, t0):
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

    def bridge_probability(self, k, k1, k0, t):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """
        p_k_to_k1 = self.conditional_probability(k1, k, t=1., t0=t)
        p_k0_to_k = self.conditional_probability(k, k0, t=t, t0=0.)
        p_k0_to_k1 = self.conditional_probability(k1, k0, t=1., t0=0.)
        return (p_k_to_k1 * p_k0_to_k) / p_k0_to_k1

    def rates(self, t, k, logits):
        t = t.squeeze()
        t1 = 1. - self.time_epsilon
        beta_integral = (t1 - t) * self.config.gamma
        wt = torch.exp(-self.vocab_size * beta_integral)
        A = 1.0
        B = (wt * self.vocab_size) / (1. - wt)
        C = wt
        qx = softmax(logits, dim=2)
        qy = torch.gather(qx, 2, k.long().unsqueeze(2))
        rate = A + B[:, None, None] * qx + C[:, None, None] * qy
        return rate