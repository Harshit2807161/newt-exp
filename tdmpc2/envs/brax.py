import numpy as np
import gymnasium as gym
import torch
from torchvision.transforms import functional as F
import jax
import jax.numpy as jnp
import logging

import brax
from brax.envs import create
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from brax.io import image

logging.getLogger("jax").setLevel(logging.WARNING)


BRAX_TASKS = {
	'brax-walker': 'walker2d',
}


class BraxWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		obs_dim = env.observation_size
		action_dim = env.action_size
		# obs_dim = env.observation_space.shape[-1]
		# action_dim = env.action_space.shape[-1]
		if cfg.obs == 'rgb':
			self.observation_space = gym.spaces.Box(
				low=0, high=255, shape=(3, cfg.render_size, cfg.render_size), dtype=np.uint8)
		else:
			self.observation_space = gym.spaces.Box(
				low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
			)
		print('Observation size:', obs_dim)
		print('Action size:', action_dim)
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(action_dim,), dtype=np.float32,
		)
		# if hasattr(self.env, '_terminate_when_unhealthy'):
		# 	self.env._terminate_when_unhealthy = False
		# 	self.env.env._terminate_when_unhealthy = False
		self._cumulative_reward = 0
		self._t = 0
		self._terminated = False
		self._rng = jax.random.PRNGKey(np.random.randint(0, 2**32 - 1))
		self._state = None
		self.max_episode_steps = 1000

	@property
	def metadata(self):
		return {}

	def _extract_info(self, info):  # expects brax info dictionary
		info = {
			'terminated': bool(info.get('episode_done', False)),
			'truncated': bool(info.get('truncation', False)),
			'success': float('nan')
		}
		info['score'] = np.clip(self._cumulative_reward, 0, 1000) / 1000
		return info

	def get_observation(self, state):
		if self.cfg.obs == 'rgb':
			return self.render()
		return np.array(state.obs[0], dtype=np.float32)

	def reset(self):
		self._rng = jax.random.PRNGKey(np.random.randint(0, 2**32 - 1))
		self._state = self.env.reset(self._rng)
		self._cumulative_reward = 0
		self._t = 0
		info = self._extract_info(self._state.info)
		self._terminated = info['terminated']
		return self.get_observation(self._state), info

	def step(self, action):
		action = jnp.array([action], dtype=jnp.float32)
		self._state = self.env.step(self._state, action)
		obs = self.get_observation(self._state)
		reward = float(self._state.reward[0])
		self._cumulative_reward += reward
		self._t += 1
		info = self._extract_info(self._state.info)
		terminated = info['terminated']
		truncated = info['truncated']
		print('Step', self._t, 'Reward:', reward, 'Cumulative:', self._cumulative_reward)
		if terminated:
			print(f'Episode terminated, length={self._t}, reward={self._cumulative_reward}, truncated={truncated}')
		return obs, reward, False, truncated, info

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def render(self, width=None, height=None):
		if self._state is None:
			raise RuntimeError("Call reset() before render().")
		w = width or self.cfg.render_size
		h = height or self.cfg.render_size
		return image.render_array(self.env.sys, self._state.pipeline_state, w, h)
	
	def close(self):
		return


def make_env(cfg):
	"""
	Make Brax environment.
	"""
	if not cfg.task in BRAX_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	env = create(
		BRAX_TASKS[cfg.task],
		episode_length=1000,
		action_repeat=2,
		auto_reset=False,
		batch_size=1,
		terminate_when_unhealthy=False,
	)
	env = BraxWrapper(env, cfg)
	return env
