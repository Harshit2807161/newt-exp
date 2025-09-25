import numpy as np
import torch
import gymnasium as gym

from dflex.envs import HopperEnv


DFLEX_TASKS = {
	"dflex-hopper": dict(
		env=HopperEnv,
		max_episode_steps=1000,
		action_repeat=2,
	),
}


class dFlexWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		if self.cfg.obs == 'rgb':
			self.observation_space = gym.spaces.Box(
				low=0, high=255, shape=(3, cfg.render_size, cfg.render_size), dtype=np.uint8)
		self.action_space = env.action_space
		self.action_repeat = DFLEX_TASKS[cfg.task]['action_repeat']
		self.max_episode_steps = DFLEX_TASKS[cfg.task]['max_episode_steps']//self.action_repeat
		self._cumulative_reward = 0

	def _extract_info(self, info):
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float('nan'),
		}
		info['score'] = info['success']
		return info

	def _np(self, x):
		if isinstance(x, np.ndarray):
			return x
		return x[0].detach().cpu().numpy()

	def reset(self, **kwargs):
		self._cumulative_reward = 0
		obs = self._np(self.env.reset())
		return obs, self._extract_info({})

	def step(self, action):
		reward = 0
		for _ in range(self.action_repeat):
			obs, _reward, truncated, info = self.env.step(torch.from_numpy(action).cuda())
			reward += _reward.clip(-5, 5).item()
		return self._np(obs), reward, False, truncated.item(), self._extract_info(info)

	@property
	def metadata(self):
		return {}
	
	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return np.zeros((8, 8, 3), dtype=np.uint8)

	def close(self):
		return


def make_env(cfg):
	"""
	Make dFlex environment.
	"""
	if cfg.task not in DFLEX_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	env = DFLEX_TASKS[cfg.task]['env'](
		render=False,
		# render=cfg.save_video,
		# logdir=f"/tmp/dflex_env_{np.random.randint(10000)}",
		num_envs=1,
		episode_length=DFLEX_TASKS[cfg.task]['max_episode_steps'],
		stochastic_init=True,
		early_termination=False,
	)
	env = dFlexWrapper(env, cfg)
	return env
