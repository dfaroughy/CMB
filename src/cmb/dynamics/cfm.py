import torch 
from dataclasses import dataclass
from cmb.dynamics.utils import OTPlanSampler

class ConditionalFlowMatching:
	''' Conditional Flow Matching base class
	'''
	def __init__(self, config: dataclass):
		self.config = config

	def define_source_target_coupling(self, batch):
		""" conditional variable z = (x_0, x1) ~ pi(x_0, x_1)
		"""		
		self.x0 = batch.source
		self.x1 = batch.target
		self.context = batch.context if self.config.DIM_CONTEXT > 0 else None
		self.mask = batch.mask	

	def sample_time(self):
		""" sample time: t ~ U[0,1]
		"""
		t = torch.rand(self.x1.shape[0], device=self.x1.device).type_as(self.x1)
		self.t = self.reshape_time(t, self.x1)

	def sample_gaussian_conditional_path(self):
		""" sample conditional path: x_t ~ p_t(x|x_0, x_1)
		"""
		mean = self.t * self.x1 + (1 - self.t) * self.x0
		std = self.config.SIGMA
		self.path = mean + std * torch.randn_like(mean)

	def conditional_vector_fields(self):
		""" conditional vector field (drift) u_t(x|x_0,x_1)
		"""
		self.drift = self.x1 - self.x0 

	def loss(self, model, batch):
		""" conditional flow-mathcing MSE loss
		"""
		self.define_source_target_coupling(batch)
		self.sample_time() 
		self.sample_gaussian_conditional_path()
		self.conditional_vector_fields()
		vt = model(x=self.path, t=self.t, context=self.context, mask=self.mask)
		ut = self.drift.to(vt.device)
		loss = torch.square(vt - ut)
		return torch.mean(loss)

	def reshape_time(self, t, x):
		if isinstance(t, (float, int)): return t
		else: return t.reshape(-1, *([1] * (x.dim() - 1)))


class OptimalTransportCFM(ConditionalFlowMatching):
	def define_source_target_coupling(self, batch):
		OT = OTPlanSampler()	
		self.x0, self.x1 = OT.sample_plan(batch.source, batch.target)
		self.context = batch.context if self.config.DIM_CONTEXT > 0 else None
		self.mask = batch.mask	

class SchrodingerBridgeCFM(ConditionalFlowMatching):
	def define_source_target_coupling(self, batch):
		regulator = 2 * self.config.SIGMA**2
		SB = OTPlanSampler(reg=regulator)
		self.x0, self.x1 = SB.sample_plan(batch.source, batch.target)	
		self.context = batch.context if self.config.DIM_CONTEXT > 0 else None
		self.mask = batch.mask	

	def sample_gaussian_conditional_path(self):
		self.mean = self.t * self.x1 + (1 - self.t) * self.x0
		std = self.config.SIGMA * torch.sqrt(self.t * (1 - self.t))
		self.path = self.mean + std * torch.randn_like(self.mean)
		
	def conditional_vector_fields(self):
		sigma_t_prime_over_sigma_t = (1 - 2 * self.t) / (2 * self.t * (1 - self.t) + 1e-8)
		self.drift = self.x1 - self.x0 + sigma_t_prime_over_sigma_t * (self.path - self.mean)