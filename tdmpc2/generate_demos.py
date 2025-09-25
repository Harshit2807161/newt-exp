import os

import torch
import hydra
from hydra.core.config_store import ConfigStore
from tensordict.tensordict import TensorDict
from torchvision.utils import make_grid, save_image

from common import set_seed
from common.buffer import Buffer
from common.world_model import WorldModel
from config import Config, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def to_td(cfg, env, obs, action=None, reward=None, value=None, terminated=None, frame=None):
	"""Creates a TensorDict for a new episode."""
	if isinstance(obs, dict):
		obs = TensorDict(obs, batch_size=(), device='cpu')
	else:
		obs = obs.cpu()
	if action is None:
		action = torch.full_like(env.rand_act(), float('nan'))
	if reward is None:
		reward = torch.tensor(float('nan')).repeat(cfg.num_envs)
	if value is None:
		value = torch.tensor(float('nan')).repeat(cfg.num_envs)
	if terminated is None:
		terminated = torch.tensor(False).repeat(cfg.num_envs)
	elif not isinstance(terminated, torch.Tensor):
		terminated = torch.stack(terminated.tolist())
	assert frame is not None, \
		'Missing frame in to_td but it is needed in demo generation.'
	td = TensorDict(
		obs=obs,
		action=action,
		reward=reward,
		value=value,
		terminated=terminated,
		frame=frame,
		batch_size=(cfg.num_envs,))
	return td


@torch.no_grad()
def estimate_value(agent, obs, action, task):
	"""Estimates the value of the current observation."""
	obs = obs.to(device='cuda', non_blocking=True)
	action = action.to(device='cuda', non_blocking=True)
	task = task.to(device='cuda', non_blocking=True)
	z = agent.model.encode(obs, task)
	value = agent.model.Q(z, action, task, return_type='avg')
	return value.cpu().squeeze(-1)


@hydra.main(version_base=None, config_name="config")
def generate_demos(cfg):
	"""Generates demonstrations."""
	assert torch.cuda.is_available()
	cfg.checkpoint = f'/data/nihansen/code/tdmpc25/checkpoints/{cfg.task}.pt'
	cfg.num_envs = 2*cfg.num_demos if cfg.task.startswith('ms') else int(2*cfg.num_demos)
	if cfg.task != 'ms-hopper-stand':
		cfg.model_size = 'B'
	cfg.save_video = True
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	assert len(cfg.tasks) == cfg.num_envs, \
		'Number of tasks must match number of environments for finetuning.'
	
	# Define environment
	env = make_env(cfg)
	tasks = torch.arange(len(cfg.tasks), dtype=torch.int32)

	# Define agent
	model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
	agent = TDMPC2(model, cfg)

	# Load checkpoint
	assert cfg.obs == 'state', \
		'Checkpoint loading only works with state observations.'
	if os.path.exists(cfg.get('checkpoint', None)):
		attempts = 0
		possible_obs_shapes = [(128,), (96,), (64,), (42,), (39,)]
		possible_action_dims = [7, 16]
		possible_task_dims = [0, 1, 512]
		while attempts < 10:
			try:
				agent.load(cfg.checkpoint)
				print(f'Loaded checkpoint from {cfg.checkpoint}.')
				break
			except Exception as e:
				print(f'Error loading checkpoint, attempt {attempts+1}/5')
				if 'weight: Tensor(shape=torch.Size([5,' in str(e) and ', 519])' in str(e):
					# Incorrect action dimension, try next possible action dim
					possible_action_dims.pop(0)
					if not possible_action_dims:
						raise ValueError(f'No valid action dimensions left to try for {cfg.checkpoint}.')
					cfg.action_dim = possible_action_dims[0]
					agent = TDMPC2(cfg)
					print('Changed action dimension to', cfg.action_dim)
				elif 'weight: Tensor(shape=torch.Size([5, 101, 512])' in str(e):
					# Incorrect task dimension, try next possible task dim
					possible_task_dims.pop(0)
					if not possible_task_dims:
						raise ValueError(f'No valid task dimensions left to try for {cfg.checkpoint}.')
					cfg.task_dim = possible_task_dims[0]
					agent = TDMPC2(cfg)
					print('Changed task dimension to', cfg.task_dim)
				elif 'size mismatch for _encoder.state.0.weight' in str(e):
					# Incorrect observation shape, try next possible obs shape
					possible_obs_shapes.pop(0)
					if not possible_obs_shapes:
						raise ValueError(f'No valid observation shapes left to try for {cfg.checkpoint}.')
					cfg.obs_shape = {'state': possible_obs_shapes[0]}
					agent = TDMPC2(cfg)
					print('Changed observation shape to', cfg.obs_shape)
				else:
					print(f'Unexpected error: {e}')
					raise e
			attempts += 1
		if attempts == 5:
			raise RuntimeError(f'Failed to load checkpoint {cfg.checkpoint} after 5 attempts.')
		is_128_dim = possible_obs_shapes[0] == (128,)
	else:
		raise ValueError(f'Checkpoint {cfg.checkpoint} does not exist.')

	# Prepare environment and metrics
	obs, info = env.reset()
	frame = info['frame']
	ep_reward = torch.zeros((cfg.num_envs,))
	ep_len = torch.ones((cfg.num_envs,), dtype=torch.int32)
	done = torch.full((cfg.num_envs,), True, dtype=torch.bool)
	tds = TensorDict({}, batch_size=(cfg.episode_length+1, cfg.num_envs), device='cpu')
	tds[0] = to_td(cfg, env, obs, frame=frame)
	frames = []

	# Prepare buffer
	cfg.buffer_size = (cfg.episode_length + 1) * cfg.num_demos
	buffer = Buffer(
		capacity=cfg.buffer_size,
		batch_size=cfg.batch_size,
		horizon=cfg.horizon,
		multiproc=False,
	)

	# Generate demos
	print(f'Generating {cfg.num_demos} demonstrations...')
	demos_collected = 0
	reward_threshold = -float('inf')
	while demos_collected < cfg.num_demos:

		# Collect experience
		if not is_128_dim:
			obs = obs[:, :possible_obs_shapes[0][0]]
		action = agent(obs, t0=done, task=tasks, eval_mode=True)
		value = estimate_value(agent, obs, action, tasks)
		obs, reward, terminated, truncated, info = env.step(action)
		assert not terminated.any(), \
			'Unexpected termination signal received.'
		ep_reward += reward
		done = terminated | truncated
	
		# Store experience
		_obs = obs.clone()
		_frame = info['frame'].clone()
		if 'final_observation' in info:
			_obs[done] = info['final_observation']
			_frame[done] = info['final_frame']
		td = to_td(cfg, env, _obs, action, reward, value, terminated, _frame)
		tds[ep_len] = td

		# Add to buffer if done and above threshold
		if done.any():
			assert done.all(), \
				'All environments must be done before adding to buffer.'
			median_reward = ep_reward.median()
			reward_threshold = max(reward_threshold, (0.75 if median_reward > 0 else 1.25) * median_reward)
			ep_success = info['final_info']['success']
			print(f'\nMean reward: {ep_reward.mean():.2f}, ')
			print(f'Median reward: {ep_reward.median():.2f}, ')
			print(f'Mean success: {ep_success.mean():.2f}, ')
			print(f'Reward threshold: {reward_threshold:.2f}')
			for i in range(cfg.num_envs):
				accept = (ep_reward[i] > reward_threshold) and \
						 (not cfg.task.startswith('mw') or ep_success[i] == 1.) and \
						 (not cfg.task.startswith('rd') or ep_success[i] == 1.) and \
						 (not cfg.task.startswith('ms') or ep_success[i] == 1. or \
							(cfg.task.startswith('ms-cartpole') or cfg.task.startswith('ms-hopper') \
							 or cfg.task.startswith('ms-ant')))
				if demos_collected >= cfg.num_demos:
					break
				elif accept:  # Accept demo
					# Add to buffer
					ep_td = tds[:, i].unsqueeze(0).clone()
					frames.append(ep_td['frame'])
					del ep_td['frame']
					demos_collected = buffer.add(ep_td)
					print(f'Added demo {demos_collected}/{cfg.num_demos} '
						  f'with reward {ep_reward[i]:.2f}, success {ep_success[i]:.2f}, and length {ep_len[i]} '
						  f'for task {cfg.tasks[i]}.')

					# Reset episode metrics
					ep_reward[i] = 0.0
					ep_len[i] = 0
				
				else:  # Reject demo
					print(f'Rejected demo for task {cfg.tasks[i]} '
						  f'with reward {ep_reward[i]:.2f} and success {ep_success[i]:.2f}.')
			
			break  # Exit regardless of number of demos collected

		else:
			ep_len += 1

	# Raise an error if not enough demos were collected
	if demos_collected < cfg.num_demos:
		print(f'[Demo collection failed] Only {demos_collected} demos collected, expected {cfg.num_demos}.')
		exit(0)

	# Save demos
	buffer.save(f'{cfg.data_dir}/{cfg.task}.pt')
	frames = torch.stack(frames, dim=0)
	frames = frames.view(torch.prod(torch.tensor(frames.shape[:3])), *frames.shape[3:])
	frames = make_grid(frames/255., nrow=frames.shape[0], padding=0)
	total_width = frames.shape[-1]
	max_width = 897_792  # Maximum width for a single image (501*8 = 4008 frames at 224 pixels each)
	if total_width < max_width:
		save_image(frames, f'{cfg.data_dir}/{cfg.task}-0.png')
	else: # Save in batches of 500k pixels
		num_batches = (total_width + max_width - 1) // max_width
		for i in range(num_batches):
			start_col = i * max_width
			end_col = min((i + 1) * max_width, total_width)
			save_image(frames[:, :, start_col:end_col], f'{cfg.data_dir}/{cfg.task}-{i}.png')
	print(f'Saved {demos_collected} demos to {cfg.data_dir}.')


if __name__ == '__main__':
	generate_demos()
