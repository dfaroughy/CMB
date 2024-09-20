import torch 
from dataclasses import dataclass
from cmb.dynamics.utils import OTPlanSampler

class ConditionalFlowMatching :
	''' Conditional Flow Matching base class
	'''
	def __init__(self, config: dataclass):
		self.config = config
		self.loss_fn = torch.nn.MSELoss(reduction='sum')

	def sample_coupling(self, batch):
		""" conditional variable z = (x_0, x1) ~ pi(x_0, x_1)
		"""		
		self.x0 = batch.source_continuous
		self.x1 = batch.target_continuous
		self.context_continuous = batch.target_context_continuous if hasattr(batch, 'target_context_continuous') else None
		self.context_discrete = batch.target_context_discrete if hasattr(batch, 'target_context_discrete') else None
		self.mask = batch.target_mask if hasattr(batch, 'target_mask') else torch.ones_like(self.x0[..., 0]).unsqueeze(-1)

	def sample_time(self):
		""" sample time: t ~ U[0,1]
		"""
		t = torch.rand(self.x1.shape[0], device=self.x1.device).type_as(self.x1)
		self.t = self.reshape_time(t, self.x1)

	def sample_bridge(self):
		""" sample conditional bridge: x_t ~ p_t(x|x_0, x_1)
		"""
		mean = self.t * self.x1 + (1. - self.t) * self.x0
		std = self.config.sigma
		self.bridge = mean + std * torch.randn_like(mean)

	def get_drift(self):
		""" conditional drift u_t(x|x_0,x_1)
		"""
		A = 0.
		B = 1.
		C = -1.
		self.drift = A * self.bridge + B * self.x1 + C * self.x0
		self.drift = self.drift * self.mask

	def loss(self, model, batch):
		""" conditional flow-mathcing MSE loss
		"""
		self.sample_coupling(batch)
		self.sample_time() 
		self.sample_bridge()
		self.get_drift()
		vt = model(t=self.t, x=self.bridge, context_continuous=self.context_continuous, context_discrete=self.context_discrete, mask=self.mask)
		ut = self.drift.to(vt.device)
		loss = self.loss_fn(vt, ut)
		return loss / self.mask.sum()

	def reshape_time(self, t, x):
		if isinstance(t, (float, int)): return t
		else: return t.reshape(-1, *([1] * (x.dim() - 1)))


class OTCFM(ConditionalFlowMatching ):
	def sample_coupling(self, batch):
		OT = OTPlanSampler()	
		self.x0, self.x1 = OT.sample_plan(batch.source_continuous, batch.target_continuous)	
		self.context_continuous = batch.target_context if hasattr(batch, 'target_context_continuous') else None
		self.context_discrete = batch.target_discrete if hasattr(batch, 'target_context_discrete') else None
		self.mask = batch.mask if hasattr(batch, 'mask') else None

class SBCFM(ConditionalFlowMatching):
	def sample_coupling(self, batch):
		regulator = 2 * self.config.sigma**2
		SB = OTPlanSampler(reg=regulator)
		self.x0, self.x1 = SB.sample_plan(batch.source_continuous, batch.target_continuous)	
		self.context_continuous = batch.target_context if hasattr(batch, 'target_context_continuous') else None
		self.context_discrete = batch.target_discrete if hasattr(batch, 'target_context_discrete') else None
		self.mask = batch.mask if hasattr(batch, 'mask') else None

	def sample_bridge(self):
		self.mean = self.t * self.x1 + (1 - self.t) * self.x0
		std = self.config.sigma * torch.sqrt(self.t * (1 - self.t))
		self.bridge = self.mean + std * torch.randn_like(self.mean)
		
	def get_drift(self):
		""" conditional drift u_t(x|x_0,x_1)
		"""
		A = (1 - 2 * self.t) / ( self.t * (1 - self.t))
		B = self.t**2 / ( self.t * (1 - self.t))
		C = -1 * (1 - self.t)**2 / ( self.t * (1 - self.t))

		self.drift = A * self.bridge + B * self.x1 + C * self.x0