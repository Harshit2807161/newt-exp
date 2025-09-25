from tensordict.tensordict import TensorDict
import torch
from torch.utils.data import DataLoader
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from common.pretrained_encoder import PretrainedEncoder
from openx_data.dataset import make_interleaved_dataset
from openx_data.oxe import make_oxe_dataset_kwargs_and_weights
from openx_data.rlds import TorchRLDSDataset


DEV_MIX = [
    # ('ucsd_kitchen_dataset_converted_externally_to_rlds', 1.0),
    ('ucsd_pick_and_place_dataset_converted_externally_to_rlds', 1.0),
	# ('austin_sailor_dataset_converted_externally_to_rlds', 1.0),
	# ('austin_sirius_dataset_converted_externally_to_rlds', 1.0),
	# ('austin_buds_dataset_converted_externally_to_rlds', 1.0),
	# ('cmu_franka_exploration_dataset_converted_externally_to_rlds', 1.0),
	# ('fractal20220817_data', 0.25),
	# ('jaco_play', 1.0),
    # ('viola', 1.0),
	# ('nyu_franka_play_dataset_converted_externally_to_rlds', 1.0),
	# ('bridge', 1.0),
	# ('taco_play', 1.0),
	# ('kuka', 0.25),
]


class OpenX():
	"""
	Data class for TD-MPC2 training. Based on the Octo dataloader.
	Reference: https://github.com/octo-models/octo
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device(cfg.rank)

		dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
			DEV_MIX,
			cfg.data_dir,
			load_camera_views=("primary",),
		)
		dataset = make_interleaved_dataset(
			dataset_kwargs_list,
			sample_weights,
			train=True,
			shuffle_buffer_size=100_000,
			batch_size=None,
			balance_weights=True,
			traj_transform_kwargs=dict(
				window_size=self.cfg.horizon+1,
				future_action_window_size=0,
				subsample_length=100,
			),
			frame_transform_kwargs=dict(
				image_augment_kwargs={
					"primary": dict(
						random_brightness=[0.1],
						random_contrast=[0.9, 1.1],
						random_saturation=[0.9, 1.1],
						random_hue=[0.05],
						augment_order=[
							"random_brightness",
							"random_contrast",
							"random_saturation",
							"random_hue",
						],
					),
				},
				resize_size=dict(
					primary=(cfg.img_size, cfg.img_size),
				),
				num_parallel_calls=128,
			),
			traj_transform_threads=16,
			traj_read_threads=16,
		)
		self._dataset = TorchRLDSDataset(
			dataset,
			prefetch_factor=1,
		)
		self._dataloader = DataLoader(
			self._dataset,
			batch_size=self.cfg.batch_size,
			num_workers=0,
		)
		self._iter = iter(self._dataloader)
		self._encoder = PretrainedEncoder(cfg)

	def _encode_image(self, image):
		"""
		Encode an image using the pretrained encoder.
		"""
		B, T, C, H, W = image.shape
		image = image.view(B*T, C, H, W)
		image = self._encoder(image)
		image = image.view(B, T, -1)
		return image

	def _to_device(self, *args, device=None):
		if device is None:
			device = self._device
		return (arg.to(device, non_blocking=True) for arg in args)

	def _permute(self, *args, dims):
		return (arg.permute(*dims) for arg in args)

	def _prepare_batch(self, data):
		"""
		Prepare a sampled batch for training (post-processing).
		"""
		image = data['observation']['image_primary'] #.permute(0, 1, 4, 2, 3)
		state = data['observation']['proprio']
		action = data['action']
		# language = data['task']['language_instruction']

		# Fake language
		language = torch.zeros(self.cfg.batch_size, self.cfg.horizon+1, 512, device=self._device, dtype=torch.float32)

		# Fake reward
		reward = torch.zeros(self.cfg.batch_size, self.cfg.horizon, 1, device=self._device, dtype=torch.float32)

		# Move to device
		image, state, action, language, reward = self._to_device(image, state, action, language, reward)

		# Plot images (debug)
		# from torchvision.utils import make_grid, save_image
		# image = image.view(-1, 3, 224, 224)
		# save_image(make_grid(image/255., nrow=4), f'/data/nihansen/code/openx/image.png')

		# Encode image
		# image = self._encode_image(image)

		# Permute
		image = image.permute(1, 0, 4, 2, 3)
		state, action, language, reward = self._permute(state, action, language, reward, dims=(1, 0, 2))

		# Convert to TensorDict
		obs = TensorDict({
			'image': image,
			'state': state,
			'language': language,
		}, batch_size=(self.cfg.horizon+1, self.cfg.batch_size))

		return obs, action[:-1], reward

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		try:
			data = next(self._iter)
		except StopIteration:
			self._iter = iter(self._dataloader)
			data = next(self._iter)

		return self._prepare_batch(data)
