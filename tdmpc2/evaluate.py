import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)
	
	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {cfg.num_global_tasks} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
		print(colored(f'Saving evaluation videos to {video_dir}', 'yellow', attrs=['bold']))
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		ep_rewards, ep_successes = [], []
		for i in range(cfg.eval_episodes // cfg.num_envs):
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), torch.tensor(False), 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done.any():
				torch.compiler.cudagraph_mark_step_begin()
				action = agent.act(obs, t0=t==0, task=task_idx)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if cfg.save_video:
					frames.append(env.render(width=cfg.render_size, height=cfg.render_size))
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if cfg.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
		ep_rewards = torch.cat(ep_rewards).mean()
		ep_successes = torch.cat(ep_successes).mean()
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))
	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
