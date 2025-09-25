from copy import deepcopy

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from common import math


class BC(torch.nn.Module):
	"""
	Behavior cloning baseline.
	"""

	def __init__(self, model, cfg):
		super().__init__()
		self.cfg = deepcopy(cfg)
		self.cfg.action_dim = cfg.action_dim
		self.device = torch.device(f'cuda:{self.cfg.rank}')
		self.model = model
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		if self.cfg.lr_schedule:
			self.scheduler = math.MultiWarmupConstantLR(
				[self.optim],
				warmup_steps=self.cfg.warmup_steps,
			)
			if self.cfg.rank == 0:
				print(f'Using {self.cfg.lr_schedule} learning rate schedule with {self.cfg.warmup_steps} warmup steps.')
		elif self.cfg.rank == 0:
			print('No learning rate schedule specified, using constant LR.')
		self.model.eval()
		if self.cfg.rank == 0:
			print('Episode length:', self.cfg.episode_length)
		if self.cfg.compile:
			self.pi = torch.compile(self._pi, mode="reduce-overhead")
			self.loss_fn = torch.compile(self._loss_fn, mode="reduce-overhead")
		else:
			self.pi = self._pi
			self.loss_fn = self._loss_fn

	def save(self, fp):
		"""Do nothing, no need to save the BC models."""
		pass

	@torch.no_grad()
	def _pi(self, obs, task=None):
		"""
		Select an action using the policy network.
		"""
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		return action, info

	@torch.no_grad()
	def forward(self, obs, t0, step, eval_mode=False, task=None, mpc=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (torch.Tensor): Whether this is the first observation in the episode.
			step (int): Current environment step.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (torch.Tensor): Task index.
			mpc (bool): Whether to use model predictive control.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		if isinstance(obs, dict):
			obs = TensorDict(obs)
		obs = obs.to(self.device, non_blocking=True)
		if task is not None and not isinstance(task, torch.Tensor):
			task = torch.tensor([task], device=self.device)
		if task is not None and task.device != self.device:
			task = task.to(self.device, non_blocking=True)
		
		action, action_info = self.pi(obs, task)
		info = TensorDict({
			"pi_mean": action_info["mean"].mean(),
			"pi_std": action_info["log_std"].exp().mean(),
		})
		return action.cpu(), info
	
	def _loss_fn(self, obs, action, task):
		"""
		Update policy using behavior cloning.
		"""
		z = self.model.encode(obs[0], task[0])
		pi_action, info = self.model.pi(z, task[0])

		# BC loss
		pi_loss = math.masked_bc_per_timestep(pi_action, action, task, self.model._action_masks).sum(0).mean()
		
		info = TensorDict({
			"pi_loss": pi_loss,
		})
		return pi_loss, info

	def _update(self, obs, action, task=None):
		# Prepare for update
		self.model.train()

		# Step the learning rate scheduler
		if self.cfg.lr_schedule:
			self.scheduler.step()

		# Compute loss
		torch.compiler.cudagraph_mark_step_begin()
		pi_loss, info = self.loss_fn(obs, action, task)

		# Update model
		pi_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Return training statistics
		self.model.eval()
		info.update({
			"grad_norm": grad_norm,
		})
		if self.cfg.lr_schedule:
			info.update({
				"lr": self.scheduler.current_lr(0, 0),
			})
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, _, task = buffer.sample(device=self.device)
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		return self._update(obs, action, **kwargs)
