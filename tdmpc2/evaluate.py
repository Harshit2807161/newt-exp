import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from config import parse_cfg
# Assuming you have an envs.py file like the example project
from envs import make_env 
from tdmpc2 import TDMPC2
import random

torch.backends.cudnn.benchmark = True

def set_seed(seed):
	"""Set seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)



@hydra.main(version_base=None, config_name="config")
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task TD-MPC2 checkpoint from your project.

	Args:
		`task`: Task name (e.g., myo-soccer-mult-control).
		`model_size`: Model size, must be one of `['B', 'L', ...]` as defined in your project.
		`checkpoint`: Path to the model checkpoint to load.
		`eval_episodes`: Number of episodes to evaluate on.
		`save_video`: Whether to save a video of the evaluation.
		`seed`: Random seed.
	
	Example usage:
	```
	$ python evaluate.py task=myo-soccer-mult-control model_size='B' checkpoint="/path/to/your/12_000_000.pt"
	```
	"""
	# Initial setup
	assert torch.cuda.is_available(), "CUDA is required for evaluation."
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	
	# Print configuration
	print(colored('='*50, 'grey'))
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	print(colored('='*50, 'grey'))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found.'
	agent.load(cfg.checkpoint)
	print(colored(f'Successfully loaded agent from {cfg.checkpoint}', 'green'))
	
	# Prepare for evaluation
	if cfg.save_video:
		video_dir = os.path.join(os.getcwd(), 'videos')
		os.makedirs(video_dir, exist_ok=True)
		print(colored(f'Saving videos to {video_dir}', 'yellow'))

	ep_rewards, ep_successes = [], []
	for i in range(cfg.eval_episodes):
		print(f"--- Starting evaluation episode {i+1}/{cfg.eval_episodes} ---")
		obs, info = env.reset()
		done, ep_reward, t = False, 0, 0
		frames = []
		if cfg.save_video:
			frames.append(env.render(width=cfg.render_size, height=cfg.render_size))

		while not done:
			# Get action from agent
			# IMPORTANT: We use mpc=True to enable planning, which is critical for performance.
			# This was the root cause of the score drop issue.
			action = agent(obs, t0=(t==0), step=0, eval_mode=True, mpc=True)
			
			# Step the environment
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated | truncated
			ep_reward += reward
			t += 1
			
			if cfg.save_video:
				frames.append(env.render(width=cfg.render_size, height=cfg.render_size))

		# Collect results
		ep_rewards.append(ep_reward)
		if 'final_info' in info and 'success' in info['final_info']:
			ep_successes.append(info['final_info']['success'].item())
		
		# Save video
		if cfg.save_video:
			video_path = os.path.join(video_dir, f'episode_{i+1}.mp4')
			imageio.mimsave(video_path, frames, fps=15)

		print(f"Episode finished in {t} steps with reward {ep_reward:.2f}")

	# Log final results
	avg_reward = np.mean(ep_rewards)
	std_reward = np.std(ep_rewards)
	
	print(colored('='*50, 'grey'))
	print(colored('Evaluation Complete!', 'green', attrs=['bold']))
	print(colored(f'Average reward: {avg_reward:.2f} ± {std_reward:.2f}', 'yellow'))

	if ep_successes:
		avg_success = np.mean(ep_successes)
		print(colored(f'Average success rate: {avg_success:.2f}', 'yellow'))
	print(colored('='*50, 'grey'))


if __name__ == '__main__':
	evaluate()