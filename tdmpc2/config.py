from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import json
import hydra
from termcolor import colored
from omegaconf import OmegaConf

from common import MODEL_SIZE, TASK_SET
from common.math import discount_heuristic


@dataclass
class Config:
	"""
	Config for experiments.
	"""

	# environment
	task: str = "soup"
	obs: str = "state"
	episodic: bool = False
	num_envs: int = 10
	env_mode: str = "async"
	tasks_fp: str = "/data/nihansen/code/tdmpc25/data/tasks.json"

	# evaluation
	checkpoint: Optional[str] = None
	eval_episodes: int = 2
	eval_freq: Optional[int] = None

	# training
	steps: int = 100_000_000
	batch_size: int = 1024
	utd: float = 0.075
	reward_coef: float = 0.1
	value_coef: float = 0.1
	consistency_coef: float = 20.0
	prior_coef: float = 10.0
	rho: float = 0.5
	lr: float = 3e-4
	enc_lr_scale: float = 0.3
	grad_clip_norm: float = 20.0
	tau: float = 0.01
	discount_denom: int = 5
	discount_min: float = 0.95
	discount_max: float = 0.995
	buffer_size: int = 10_000_000
	use_demos: bool = True
	no_demo_buffer: bool = False
	demo_steps: int = 200_000
	lr_schedule: Optional[str] = None
	warmup_steps: int = 5_000
	seeding_coef: int = 5
	exp_name: str = "default"
	finetune: bool = False

	# planning
	mpc: bool = True
	iterations: int = 6
	num_samples: int = 512
	num_elites: int = 64
	num_pi_trajs: int = 24
	horizon: int = 3
	min_std: float = 0.05
	max_std: float = 2.0
	temperature: float = 0.5
	constrained_planning: bool = True
	constraint_start_step: int = 2_000_000
	constraint_final_step: int = 10_000_000
	constraint_min_weight: float = 0.0

	# actor
	log_std_min: float = -10
	log_std_max: float = 2.0
	entropy_coef: float = 1e-4
	use_scaled_entropy: bool = True

	# critic
	num_bins: int = 101
	vmin: float = -10.0
	vmax: float = +10.0

	# architecture
	model_size: Optional[str] = None
	num_channels: int = 32
	num_enc_layers: int = 3
	enc_dim: int = 1024
	mlp_dim: int = 1024
	latent_dim: int = 512
	task_dim: int = 512
	num_q: int = 5
	simnorm_dim: int = 8
	disable_task_emb: bool = False

	# bc baseline
	bc_baseline: bool = False

	# logging
	wandb_project: str = "tdmpc3"
	wandb_entity: str = "nicklashansen"
	wandb_silent: bool = False
	enable_wandb: bool = True

	# misc
	multiproc: bool = False
	rank: int = 0
	world_size: int = 1
	port: Optional[str] = None
	compile: bool = True
	save_video: bool = False
	render_size: int = 224
	save_agent: bool = True
	save_freq: Optional[int] = None
	save_buffer: bool = False
	data_dir: str = "/data/nihansen/code/tdmpc25/data"
	seed: int = 1

	# convenience (filled at runtime)
	work_dir: Optional[str] = None
	task_title: Optional[str] = None
	tasks: Any = None
	global_tasks: Any = None
	num_tasks: Optional[int] = None
	num_global_tasks: Optional[int] = None
	task_embeddings: Any = None
	obs_shape: Any = None
	action_dim: Optional[int] = None
	episode_length: Optional[int] = None
	obs_shapes: Any = None
	action_dims: Any = None
	episode_lengths: Any = None
	discounts: Any = None
	bin_size: Optional[float] = None
	child_env: bool = False

	get = lambda self, val, default=None: getattr(self, val, default)


def split_by_rank(global_list, rank, world_size):
	"""Split a global list into sublists for each rank."""
	return [global_list[i] for i in range(len(global_list)) if i % world_size == rank]


def parse_cfg(cfg):
	"""
	Parses the experiment config dataclass. Mostly for convenience.
	"""
	# Convenience
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1)  # Bin size for discrete regression

	# Model size
	if cfg.task != 'soup' and not cfg.finetune and cfg.model_size is None:
		cfg.model_size = 'B'
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v

	# Set defaults
	cfg.tasks = TASK_SET.get(cfg.task, [cfg.task] * cfg.num_envs)
	cfg.num_tasks = len(dict.fromkeys(cfg.tasks))  # Unique tasks
	cfg.global_tasks = deepcopy(cfg.tasks)
	cfg.num_global_tasks = cfg.num_tasks
	if cfg.task == 'soup':
		cfg.num_envs = cfg.num_tasks
		print(colored(f'Number of tasks in soup: {cfg.num_global_tasks}', 'green', attrs=['bold']))
	cfg.eval_freq = 20 * 500 * cfg.num_envs
	cfg.save_freq = 5 * cfg.eval_freq
	if cfg.use_demos and cfg.checkpoint is None:
		# Warmup LR when pretraining
		cfg.lr_schedule = "warmup"
	if not (cfg.task == 'soup' and cfg.use_demos and cfg.demo_steps > 0 and cfg.num_pi_trajs > 0):
		cfg.constrained_planning = False
	if cfg.finetune:
		assert cfg.checkpoint is not None, "Checkpoint must be specified for finetuning."
		cfg.steps = 1 #100_000
		cfg.buffer_size = 100_000
		cfg.eval_episodes = 1 #10 if cfg.num_envs == 1 else 1
		cfg.use_demos = False
		cfg.env_mode = 'sync'
		cfg.save_video = False #True
	elif cfg.bc_baseline and cfg.task != 'soup':
		cfg.task_dim = 0
		cfg.prior_coef = 0
		cfg.steps = 1
		cfg.horizon = 2
		cfg.buffer_size = 50_000
		cfg.eval_episodes = 2
		cfg.env_mode = 'sync'
		cfg.save_video = False
		cfg.save_agent = False
		cfg.demo_steps = 20_000
	elif cfg.task != 'soup':
		cfg.task_dim = 0
		cfg.prior_coef = 0
		cfg.steps = 10_000_000 if cfg.task.startswith('atari-') else 5_000_000
		cfg.buffer_size = 1_000_000
		cfg.eval_episodes = 1
		cfg.save_freq *= 2
		cfg.use_demos = False
		cfg.env_mode = 'sync'
		cfg.save_video = True

	# Load task embeddings
	with open(cfg.tasks_fp, "r") as f:
		task_info = json.load(f)
	cfg.task_embeddings = []
	cfg.episode_lengths = []
	cfg.discounts = []
	cfg.action_dims = []
	for task in cfg.tasks:
		assert task in task_info, f'Task {task} not found in task embeddings.'
		cfg.task_embeddings.append(task_info[task]['text_embedding'])
		cfg.episode_lengths.append(task_info[task]['max_episode_steps'])
		if 'discount_factor' in task_info[task]:
			cfg.discounts.append(task_info[task]['discount_factor'])
		else:
			cfg.discounts.append(discount_heuristic(cfg, task_info[task]['max_episode_steps']))
		cfg.action_dims.append(task_info[task]['action_dim'])

	return OmegaConf.to_object(cfg)
