import numpy as np
# import gymnasium as gym
from envs.wrappers.timeout import Timeout
import gym
import warnings
from envs.wrappers.action_bundler import ActionSpaceWrapper
from myosuite.utils.quat_math import quat2euler_intrinsic

MYOSUITE_TASKS = {
	'myo-reach': 'myoHandReachFixed-v0',
	'myo-reach-hard': 'myoHandReachRandom-v0',
	'myo-pose': 'myoHandPoseFixed-v0',
	'myo-pose-hard': 'myoHandPoseRandom-v0',
	'myo-obj-hold': 'myoHandObjHoldFixed-v0',
	'myo-obj-hold-hard': 'myoHandObjHoldRandom-v0',
	'myo-key-turn': 'myoHandKeyTurnFixed-v0',
	'myo-key-turn-hard': 'myoHandKeyTurnRandom-v0',
	'myo-pen-twirl': 'myoHandPenTwirlFixed-v0',
	'myo-pen-twirl-hard': 'myoHandPenTwirlRandom-v0',
	'myo-soccer':'myoChallengeSoccerP1-v0',
	'last3':'myoChallengeSoccerP1-v0',
	'myoChallengeSoccerP1-v0':'myoChallengeSoccerP1-v0',
	'myo-soccer-hard':'myoChallengeSoccerP2-v0',
	'myo-locomotion': 'myoChallengeOslRunFixed-v0'
}

class MyoSuiteWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.camera_id = 'goal_view'
		self.action_repeat = 4
		self.torso_joints = [
    'Abs_r3', 'Abs_t1', 'Abs_t2',
    'L1_L2_AR', 'L1_L2_FE', 'L1_L2_LB',
    'L2_L3_AR', 'L2_L3_FE', 'L2_L3_LB',
    'L3_L4_AR', 'L3_L4_FE', 'L3_L4_LB',
    'L4_L5_AR', 'L4_L5_FE', 'L4_L5_LB',
    'axial_rotation', 'flex_extension', 'lat_bending']

		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=self.get_obs().shape, dtype=np.float32
		)
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(self.env.action_space.shape[0],), dtype=np.float32
		)
		self.torso_qpos_indices = [
            self.env.sim.model.joint(jnt).qposadr[0] for jnt in self.torso_joints
        ]


	def reset(self):
		obs, info = self.env.reset()
		obs = self.get_obs()
		info = self._extract_info(info)
		return obs, info

	def _extract_info(self, info):
		success = self._goal_scored_condition()
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float(success),
		}
		info['score'] = info['success']
		return info

	def get_obs(self):
		obs = self.env.get_obs_dict(self.env.sim)
		obs = [obs[key] for key in sorted(obs.keys())]
		return np.concatenate(obs)

	def step(self, action):
		self.ct = 0
		reward = 0
		for _ in range(self.action_repeat):
			obs, r, terminated, truncated, info = self.env.step(action.copy())
			# return obs, reward, False, info
			reward += self._get_reward_mult()
			self.ct+=1
			# print(reward)
		info['success'] = info['solved']
		info['terminated'] = terminated
		info['truncated'] = truncated
		del info['obs_dict']
		del info['visual_dict']
		del info['proprio_dict']
		del info['rwd_dict']
		del info['state']
		# print(self.ct)
		return self.get_obs(), reward, False, False, info


	def _get_cyclic_reward(self, obs):
		"""
		Return a bounded reward in [0,1] that measures how close hip angles are to a desired cyclic pattern.
		Uses self.steps and self.hip_period (as in your snippet).
		"""
		# compute desired phase angles exactly like your function
		phase_var = (float(getattr(self, "steps", 0)) / float(getattr(self, "hip_period", 60))) % 1.0
		des_angles = np.array([
			0.8 * np.cos(phase_var * 2 * np.pi + np.pi),
			0.8 * np.cos(phase_var * 2 * np.pi),
		], dtype=np.float32)

		# read current hip flexion angles (assumes _get_angle exists and returns [l, r])
		angles = np.array(self._get_angle(["hip_flexion_l", "hip_flexion_r"]), dtype=np.float32)
		err = np.linalg.norm(des_angles - angles)  # L2 error

		# convert error -> reward in (0,1], using an RBF-like mapping
		alpha = 6.0   # sharpness; tune (larger -> reward concentrated near perfect match)
		r_cyclic = float(np.exp(-alpha * err))  # near 1 when err~0, decays quickly

		return r_cyclic
	
	def _get_angle(self, names):
		"""
		Get the angles of a list of named joints.
        """
		return np.array(
            [
                self.sim.data.qpos[
                    self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]
                ]
                for name in names
            ]
        )

	def _get_reward_mult_og(self):
		"""Returns a reward to the agent."""
		_STAND_HEIGHT = self.cfg.stand_height
		_MOVE_SPEED = self.cfg.move_speed
		obs = self.env.get_obs_dict(self.env.sim)
		h = self.env.sim.data.site('head').xpos[2]
		# print(h)
		ball_pos = np.array(obs['ball_pos']).copy()        # shape (3,)
		root_qpos = np.array(obs['model_root_pos']).copy() # root_qpos[0:3] -> x,y,z
		root_xy = root_qpos[0:2].copy()
		root_z = float(root_qpos[2])
		z_target = 0.9
		r_posture = self._tolerance(root_z, bounds=(z_target, z_target+0.03), margin=0.2)
		standing = self._tolerance(h,
									bounds=(_STAND_HEIGHT, _STAND_HEIGHT+0.045),
									margin=_STAND_HEIGHT/4)
		stand_reward = r_posture * r_posture
		act = self.env.sim.data.act[:].copy()
		act_mag = np.mean(np.square(act)) if self.env.sim.model.na !=0 else 0
		small_control = self._tolerance(act_mag, margin=1,
										value_at_margin=0,
										sigmoid='quadratic')
		small_control = (4 + small_control) / 5
		small_control = 1
		if_goal_scored = self._goal_scored_condition() 
		# if_goal_scored = 1
		goal_score = 1		
		if if_goal_scored:
			goal_score = 10
		ball_pos = np.array(obs['ball_pos']).copy()        
		r_toe = np.array(obs['r_toe_pos'])[0:2].copy()
		l_toe = np.array(obs['l_toe_pos'])[0:2].copy()

		# toe-to-ball horizontal distance (min of two toes)
		d_r = np.linalg.norm(r_toe - ball_pos[0:2])
		d_l = np.linalg.norm(l_toe - ball_pos[0:2])
		d_toe = min(d_r, d_l)
		toe_ball_rew = self._tolerance(d_toe, margin=0.75, value_at_margin=0.5, sigmoid='quadratic')
		toe_ball_rew = (toe_ball_rew + 3)/4
		fallen = self._get_fallen_condition()
		# pain = self.get_jnt_limit_violation() # Joint limit violation torque as pain score
		pos = self.env.sim.data.joint('root').qpos.copy()
		y_pos = pos[1]
		straightness = self._tolerance(abs(y_pos), bounds=(-0.3, 0.3), margin=0.1, sigmoid='quadratic')
		straightness = (3 + straightness) / 4
		goal_distance,ball_distance,h_ball = self.get_ball_rel_reward(obs)
		com_vel = self.sim.data.joint('root').qvel.copy()
		horizontal_velocity = com_vel[0]
		move = self._tolerance(horizontal_velocity,
							bounds=(_MOVE_SPEED, float('inf')),
							margin=_MOVE_SPEED, value_at_margin=0,
							sigmoid='linear')
		move = (5*move + 1) / 6
		# if(ball_distance<0.2):
		# 	move = max(1 - (horizontal_velocity/5),0)
		# init_posture_rew = self._get_torso_posture_reward(obs)
		# init_posture_rew = (init_posture_rew + 4)/5
		ball_to_goal = 1
		if(ball_distance<0.2 and stand_reward>=0.75):
			ball_to_goal = self._tolerance(goal_distance,
							margin=5, value_at_margin=0.8,
							sigmoid='quadratic')
			ball_to_goal = (1+ 2*(ball_to_goal))
			if ball_distance<0.15:
				move = max(1 - abs(horizontal_velocity) / 2.0, 0)
			# slow = max(1 - abs(horizontal_velocity) / 1.0, 0)  
			# move = 0.3 * move + 0.7 * slow   
		# grf_reward = self.foot_grf_rew(obs,speed=horizontal_velocity)
		# grf_reward = (grf_reward + 4)/5
		# ball_h_rew = self._tolerance(h_ball, bounds=(0.1, 0.2),
		# 					margin=0.05, value_at_margin=0.72,
		# 					sigmoid='quadratic')
		# ball_h_rew = (ball_h_rew + 4)/5
		# if h_ball<0.2 and h_ball>0.1:
		# 	ball_h_rew = 10*h_ball
		# if h_ball>0.2:
		# 	ball_h_rew = 1e-3
		rew_components = dict(
			small_control=small_control,
			goal_score = goal_score,
			stand_reward=stand_reward,
			move=move,
			straightness=straightness,
			toe_ball_rew=toe_ball_rew,
			# init_posture_rew=init_posture_rew,
			# grf_reward=grf_reward,
			# ball_h_rew=ball_h_rew,
			ball_to_goal=ball_to_goal
		)
		# print(init_posture_rew)
		# pain = self.get_jnt_limit_violation()
		# print()
		# print(quaternion_to_axis_angle(obs['torso_angle']))
		# print(obs['torso_angle'])
		# print("horizontal_velocity: ", horizontal_velocity)
		# print("height:", self.env.sim.data.site('head').xpos[2], "standing:", standing)
		# print(small_control * stand_reward * move, "small_control:", small_control, "stand_reward:", stand_reward, "move:", move)
		# return goal_score * small_control * stand_reward * move * straightness * ball_to_goal * ball_h_rew# * (1 - self._get_fallen_condition())
		# return small_control * stand_reward * move * straightness * init_posture_rew * grf_reward
		final_reward = 1
		for k, v in rew_components.items():
			final_reward *= v
		# final_reward = goal_score * small_control * stand_reward * move * straightness * ball_to_goal * ball_h_rew
		# final_reward = small_control * stand_reward * move * straightness
		final_reward = goal_score * small_control * stand_reward * move * toe_ball_rew * ball_to_goal
		# final_reward = small_control * stand_reward * move * straightness * init_posture_rew 
		return final_reward



	# def _get_reward_mult(self):
	# 	"""Returns a reward to the agent."""
	# 	_STAND_HEIGHT = self.cfg.stand_height
	# 	_MOVE_SPEED = self.cfg.move_speed

	# 	obs = self.env.get_obs_dict(self.env.sim)
	# 	t = float(obs['time'].item()) if hasattr(obs['time'], 'item') else float(obs['time'])
	# 	# bookkeeping for dt and previous ball pos
	# 	prev_t = getattr(self, "_prev_time", t)
	# 	dt = max(1e-6, t - prev_t)
	# 	prev_ball = getattr(self, "_prev_ball_pos", obs['ball_pos'].copy())

		
	# 	h = self.env.sim.data.site('head').xpos[2]
	# 	# print(h)
	# 	ball_pos = np.array(obs['ball_pos']).copy()        # shape (3,)
	# 	root_qpos = np.array(obs['model_root_pos']).copy() # root_qpos[0:3] -> x,y,z
	# 	root_xy = root_qpos[0:2].copy()
	# 	root_z = float(root_qpos[2])
		
	# 	pitch, uprightness = self.get_torso_uprightness(obs)
	# 	upright = self._tolerance(uprightness, bounds=(0.95, 1), margin=0.3)
	# 	# print(pitch,uprightness,upright)
	# 	z_target = 0.9
	# 	r_posture = self._tolerance(root_z, bounds=(z_target, z_target+0.03), margin=0.25)
	# 	standing = self._tolerance(h,
	# 								bounds=(_STAND_HEIGHT, _STAND_HEIGHT+0.045),
	# 								margin=_STAND_HEIGHT/4,value_at_margin=0.2,sigmoid="linear")
	# 	posture_deviation = self._get_torso_posture_reward(obs)
	# 	init_posture_rew = self._tolerance(posture_deviation,
	# 						margin=1.5, value_at_margin=0.5,
	# 						sigmoid='linear')
	# 	stand_reward = r_posture * init_posture_rew
	# 	off = (abs(z_target-root_z)/z_target)*100
	# 	# print(stand_reward,off)
	# 	act = self.env.sim.data.act[:].copy()
	# 	act_mag = np.mean(np.square(act)) if self.env.sim.model.na !=0 else 0
	# 	small_control = self._tolerance(act_mag, margin=1,
	# 									value_at_margin=0,
	# 									sigmoid='quadratic')
	# 	small_control = (4 + small_control) / 5
	# 	small_control = 1
	# 	if_goal_scored = self._goal_scored_condition() 
	# 	# if_goal_scored = 1
	# 	goal_score = 1		
	# 	if if_goal_scored:
	# 		goal_score = 10
	# 	ball_pos = np.array(obs['ball_pos']).copy()        
	# 	r_toe = np.array(obs['r_toe_pos'])[0:2].copy()
	# 	l_toe = np.array(obs['l_toe_pos'])[0:2].copy()

	# 	# toe-to-ball horizontal distance (min of two toes)
	# 	d_r = np.linalg.norm(r_toe - ball_pos[0:2])
	# 	d_l = np.linalg.norm(l_toe - ball_pos[0:2])
	# 	d_toe = min(d_r, d_l)
	# 	toe_ball_rew = self._tolerance(d_toe, margin=0.75, value_at_margin=0.5, sigmoid='quadratic')
	# 	toe_ball_rew = (toe_ball_rew + 3)/4
	# 	fallen = self._get_fallen_condition()
	# 	# pain = self.get_jnt_limit_violation() # Joint limit violation torque as pain score
	# 	pos = self.env.sim.data.joint('root').qpos.copy()
	# 	y_pos = pos[1]
	# 	straightness = self._tolerance(abs(y_pos), bounds=(-0.3, 0.3), margin=0.1, sigmoid='quadratic')
	# 	straightness = (3 + straightness) / 4
	# 	goal_distance,ball_distance,h_ball = self.get_ball_rel_reward(obs)
	# 	com_vel = self.sim.data.joint('root').qvel.copy()
	# 	horizontal_velocity = com_vel[0]

	# 	ball_vel = (ball_pos - prev_ball) / dt
	# 	v_toward_goal = float(ball_vel[0])  
	# 	vel_reward = 1

	# 	move = self._tolerance(horizontal_velocity,
	# 						bounds=(_MOVE_SPEED, float('inf')),
	# 						margin=_MOVE_SPEED, value_at_margin=0,
	# 						sigmoid='linear')
	# 	move = (5*move + 1) / 6
	# 	# if(ball_distance<0.2):
	# 	# 	move = max(1 - (horizontal_velocity/5),0)
	# 	# init_posture_rew = (init_posture_rew + 4)/5
	# 	ball_to_goal = 1
	# 	if(ball_distance<0.2 and stand_reward>=0.75):
	# 		ball_to_goal = self._tolerance(goal_distance,
	# 						margin=5, value_at_margin=0.8,
	# 						sigmoid='quadratic')
	# 		ball_to_goal = (1+ 2*(ball_to_goal))
	# 		if ball_distance<0.15:
	# 			move = max(1 - abs(horizontal_velocity) / 2.0, 0)
	# 		if ball_distance<0.10:	
	# 			toe_ball_rew = 1
	# 			vel_reward = self._tolerance(v_toward_goal,
	# 						bounds=(10.0,20.0),
	# 						margin=5, value_at_margin=0.5,
	# 						sigmoid='linear')
	# 			vel_reward = 1 + vel_reward

	# 		# slow = max(1 - abs(horizontal_velocity) / 1.0, 0)  
	# 		# move = 0.3 * move + 0.7 * slow   
	# 	# grf_reward = self.foot_grf_rew(obs,speed=horizontal_velocity)
	# 	# grf_reward = (grf_reward + 4)/5
	# 	# ball_h_rew = self._tolerance(h_ball, bounds=(0.1, 0.2),
	# 	# 					margin=0.05, value_at_margin=0.72,
	# 	# 					sigmoid='quadratic')
	# 	# ball_h_rew = (ball_h_rew + 4)/5
	# 	# if h_ball<0.2 and h_ball>0.1:
	# 	# 	ball_h_rew = 10*h_ball
	# 	# if h_ball>0.2:
	# 	# 	ball_h_rew = 1e-3
	# 	rew_components = dict(
	# 		small_control=small_control,
	# 		goal_score = goal_score,
	# 		stand_reward=stand_reward,
	# 		standing = standing,
	# 		move=move,
	# 		straightness=straightness,
	# 		toe_ball_rew=toe_ball_rew,
	# 		uprightness=uprightness,
	# 		init_posture_rew=init_posture_rew,
	# 		# grf_reward=grf_reward,
	# 		# ball_h_rew=ball_h_rew,
	# 		ball_to_goal=ball_to_goal
	# 	)
	# 	# print(init_posture_rew)
	# 	# pain = self.get_jnt_limit_violation()
	# 	# print()
	# 	# print(quaternion_to_axis_angle(obs['torso_angle']))
	# 	# print(obs['torso_angle'])
	# 	# print("horizontal_velocity: ", horizontal_velocity)
	# 	# print("height:", self.env.sim.data.site('head').xpos[2], "standing:", standing)
	# 	# print(small_control * stand_reward * move, "small_control:", small_control, "stand_reward:", stand_reward, "move:", move)
	# 	# return goal_score * small_control * stand_reward * move * straightness * ball_to_goal * ball_h_rew# * (1 - self._get_fallen_condition())
	# 	# return small_control * stand_reward * move * straightness * init_posture_rew * grf_reward

	# 	self._prev_ball_pos = ball_pos.copy()
	# 	self._prev_time = t

		
	# 	final_reward = 1
	# 	for k, v in rew_components.items():
	# 		final_reward *= v
	# 	# final_reward = goal_score * small_control * stand_reward * move * straightness * ball_to_goal * ball_h_rew
	# 	# final_reward = small_control * stand_reward * move * straightness
	# 	final_reward = goal_score * small_control * stand_reward * move * toe_ball_rew * ball_to_goal * vel_reward
	# 	# final_reward = small_control * stand_reward * move * straightness * init_posture_rew 
	# 	return final_reward, rew_components


	# def _get_reward_mult(self):
	# 	"""Returns a reward to the agent."""
	# 	_STAND_HEIGHT = self.cfg.stand_height
	# 	_MOVE_SPEED = self.cfg.move_speed

	# 	obs = self.env.get_obs_dict(self.env.sim)
	# 	t = float(obs['time'].item()) if hasattr(obs['time'], 'item') else float(obs['time'])
	# 	# bookkeeping for dt and previous ball pos
	# 	prev_t = getattr(self, "_prev_time", t)
	# 	dt = max(1e-6, t - prev_t)
	# 	prev_ball = getattr(self, "_prev_ball_pos", obs['ball_pos'].copy())
	# 	h = self.env.sim.data.site('head').xpos[2]
	# 	# print(h)
	# 	ball_pos = np.array(obs['ball_pos']).copy()        # shape (3,)
	# 	root_qpos = np.array(obs['model_root_pos']).copy() # root_qpos[0:3] -> x,y,z
	# 	root_xy = root_qpos[0:2].copy()
	# 	root_z = float(root_qpos[2])
		
	# 	# pitch, uprightness = self.get_torso_uprightness(obs)
	# 	# upright = self._tolerance(uprightness, bounds=(0.95, 1), margin=0.3)
	# 	# print(pitch,uprightness,upright)
	# 	z_target = 0.9
	# 	r_posture = self._tolerance(root_z, bounds=(z_target, z_target+0.03), margin=0.25)
	# 	# standing = self._tolerance(h,
	# 	# 							bounds=(_STAND_HEIGHT, _STAND_HEIGHT+0.045),
	# 	# 							margin=_STAND_HEIGHT/4)
	# 	standing = self._tolerance(h,
	# 								bounds=(_STAND_HEIGHT, _STAND_HEIGHT+0.045),
	# 								margin=_STAND_HEIGHT/4,value_at_margin=0.2,sigmoid="linear")
	# 	posture_deviation = self._get_torso_posture_reward(obs)
	# 	init_posture_rew = self._tolerance(posture_deviation,
	# 						margin=1.5, value_at_margin=0.5,
	# 						sigmoid='linear')
	# 	stand_reward = r_posture * standing
	# 	act = self.env.sim.data.act[:].copy()
	# 	act_mag = np.mean(np.square(act)) if self.env.sim.model.na !=0 else 0
	# 	small_control = self._tolerance(act_mag, margin=1,
	# 									value_at_margin=0,
	# 									sigmoid='quadratic')
	# 	small_control = (4 + small_control) / 5
	# 	# small_control = 1
	# 	if_goal_scored = self._goal_scored_condition() 
	# 	# if_goal_scored = 1
	# 	goal_score = 1		
	# 	if if_goal_scored:
	# 		goal_score = 10
	# 	ball_pos = np.array(obs['ball_pos']).copy()        
	# 	r_toe = np.array(obs['r_toe_pos'])[0:2].copy()
	# 	l_toe = np.array(obs['l_toe_pos'])[0:2].copy()

	# 	# toe-to-ball horizontal distance (min of two toes)
	# 	d_r = np.linalg.norm(r_toe - ball_pos[0:2])
	# 	d_l = np.linalg.norm(l_toe - ball_pos[0:2])
	# 	d_toe = min(d_r, d_l)
	# 	toe_ball_rew = self._tolerance(d_toe, margin=0.75, value_at_margin=0.5, sigmoid='quadratic')
	# 	toe_ball_rew = (toe_ball_rew + 3)/4
	# 	fallen = self._get_fallen_condition()
	# 	# pain = self.get_jnt_limit_violation() # Joint limit violation torque as pain score
	# 	pos = self.env.sim.data.joint('root').qpos.copy()
	# 	y_pos = pos[1]
	# 	straightness = self._tolerance(abs(y_pos), bounds=(-0.3, 0.3), margin=0.1, sigmoid='quadratic')
	# 	straightness = (3 + straightness) / 4
	# 	goal_distance,ball_distance,h_ball = self.get_ball_rel_reward(obs)
	# 	com_vel = self.sim.data.joint('root').qvel.copy()
	# 	horizontal_velocity = com_vel[0]

	# 	ball_vel = (ball_pos - prev_ball) / dt
	# 	v_toward_goal = float(ball_vel[0])  
	# 	vel_reward = 1

	# 	move = self._tolerance(horizontal_velocity,
	# 						bounds=(_MOVE_SPEED, float('inf')),
	# 						margin=_MOVE_SPEED, value_at_margin=0,
	# 						sigmoid='linear')
	# 	move = (5*move + 1) / 6
	# 	# if(ball_distance<0.2):
	# 	# 	move = max(1 - (horizontal_velocity/5),0)
	# 	# init_posture_rew = (init_posture_rew + 4)/5
	# 	ball_to_goal = 1
	# 	if(ball_distance<0.2 and stand_reward>=0.75):
	# 		ball_to_goal = self._tolerance(goal_distance,
	# 						margin=5, value_at_margin=0.8,
	# 						sigmoid='quadratic')
	# 		ball_to_goal = (1+ 2*(ball_to_goal))
	# 		if ball_distance<0.15:
	# 			move = max(1 - abs(horizontal_velocity) / 2.0, 0)
	# 		if ball_distance<0.10:
	# 			toe_ball_rew = 1
	# 			vel_reward = self._tolerance(v_toward_goal,
	# 						bounds=(10.0,20.0),
	# 						margin=5, value_at_margin=0.5,
	# 						sigmoid='linear')
	# 			vel_reward = 1 + vel_reward
	# 		# slow = max(1 - abs(horizontal_velocity) / 1.0, 0)  
	# 		# move = 0.3 * move + 0.7 * slow   
	# 	# grf_reward = self.foot_grf_rew(obs,speed=horizontal_velocity)
	# 	# grf_reward = (grf_reward + 4)/5
	# 	# ball_h_rew = self._tolerance(h_ball, bounds=(0.1, 0.2),
	# 	# 					margin=0.05, value_at_margin=0.72,
	# 	# 					sigmoid='quadratic')
	# 	# ball_h_rew = (ball_h_rew + 4)/5
	# 	# if h_ball<0.2 and h_ball>0.1:
	# 	# 	ball_h_rew = 10*h_ball
	# 	# if h_ball>0.2:
	# 	# 	ball_h_rew = 1e-3
	# 	rew_components = dict(
	# 		small_control=small_control,
	# 		goal_score = goal_score,
	# 		stand_reward=stand_reward,
	# 		move=move,
	# 		straightness=straightness,
	# 		toe_ball_rew=toe_ball_rew,
	# 		init_posture_rew=init_posture_rew,
	# 		# grf_reward=grf_reward,
	# 		# ball_h_rew=ball_h_rew,
	# 		ball_to_goal=ball_to_goal
	# 	)
	# 	# print(init_posture_rew)
	# 	# pain = self.get_jnt_limit_violation()
	# 	# print()
	# 	# print(quaternion_to_axis_angle(obs['torso_angle']))
	# 	# print(obs['torso_angle'])
	# 	# print("horizontal_velocity: ", horizontal_velocity)
	# 	# print("height:", self.env.sim.data.site('head').xpos[2], "standing:", standing)
	# 	# print(small_control * stand_reward * move, "small_control:", small_control, "stand_reward:", stand_reward, "move:", move)
	# 	# return goal_score * small_control * stand_reward * move * straightness * ball_to_goal * ball_h_rew# * (1 - self._get_fallen_condition())
	# 	# return small_control * stand_reward * move * straightness * init_posture_rew * grf_reward

	# 	self._prev_ball_pos = ball_pos.copy()
	# 	self._prev_time = t
	# 	final_reward = 1
	# 	# for k, v in rew_components.items():
	# 	# 	final_reward *= v
	# 	# final_reward = goal_score * small_control * stand_reward * move * straightness * ball_to_goal * ball_h_rew
	# 	# final_reward = small_control * stand_reward * move * straightness
	# 	final_reward = goal_score * small_control * stand_reward * move * toe_ball_rew * ball_to_goal * vel_reward
	# 	# final_reward = small_control * stand_reward * move * straightness * init_posture_rew 
	# 	return final_reward, rew_components


	def _get_reward_mult(self):
		"""Returns a reward to the agent."""
		_STAND_HEIGHT = self.cfg.stand_height
		_MOVE_SPEED = self.cfg.move_speed

		obs = self.env.get_obs_dict(self.env.sim)
		t = float(obs['time'].item()) if hasattr(obs['time'], 'item') else float(obs['time'])
		# bookkeeping for dt and previous ball pos
		prev_t = getattr(self, "_prev_time", t)
		dt = max(1e-6, t - prev_t)
		prev_ball = getattr(self, "_prev_ball_pos", obs['ball_pos'].copy())
		h = self.env.sim.data.site('head').xpos[2]
		# print(h)
		ball_pos = np.array(obs['ball_pos']).copy()        # shape (3,)
		root_qpos = np.array(obs['model_root_pos']).copy() # root_qpos[0:3] -> x,y,z
		root_xy = root_qpos[0:2].copy()
		root_z = float(root_qpos[2])
		
		# pitch, uprightness = self.get_torso_uprightness(obs)
		# upright = self._tolerance(uprightness, bounds=(0.95, 1), margin=0.3)
		# print(pitch,uprightness,upright)
		z_target = 0.9
		r_posture = self._tolerance(root_z, bounds=(z_target, z_target+0.03), margin=0.25)
		# standing = self._tolerance(h,
		# 							bounds=(_STAND_HEIGHT, _STAND_HEIGHT+0.045),
		# 							margin=_STAND_HEIGHT/4)
		standing = self._tolerance(h,
									bounds=(_STAND_HEIGHT, _STAND_HEIGHT+0.045),
									margin=_STAND_HEIGHT/4,value_at_margin=0.2,sigmoid="linear")
		posture_deviation = self._get_torso_posture_reward(obs)
		init_posture_rew = self._tolerance(posture_deviation,
							margin=1.5, value_at_margin=0.5,
							sigmoid='linear')
		stand_reward = r_posture * standing
		act = self.env.sim.data.act[:].copy()
		act_mag = np.mean(np.square(act)) if self.env.sim.model.na !=0 else 0
		small_control = self._tolerance(act_mag, margin=1,
										value_at_margin=0,
										sigmoid='quadratic')
		small_control = (4 + small_control) / 5
		# small_control = 1
		if_goal_scored = self._goal_scored_condition() 
		# if_goal_scored = 1
		goal_score = 1		
		if if_goal_scored:
			goal_score = 75
		ball_pos = np.array(obs['ball_pos']).copy()        
		r_toe = np.array(obs['r_toe_pos'])[0:2].copy()
		l_toe = np.array(obs['l_toe_pos'])[0:2].copy()
		# toe-to-ball horizontal distance (min of two toes)
		d_r = np.linalg.norm(r_toe - ball_pos[0:2])
		d_l = np.linalg.norm(l_toe - ball_pos[0:2])
		d_toe = min(d_r, d_l)
		toe_ball_rew = self._tolerance(d_toe, margin=0.75, value_at_margin=0.5, sigmoid='quadratic')
		toe_ball_rew = (toe_ball_rew + 3)/4
		fallen = self._get_fallen_condition()
		# pain = self.get_jnt_limit_violation() # Joint limit violation torque as pain score
		pos = self.env.sim.data.joint('root').qpos.copy()
		y_pos = pos[1]
		# straightness = self._tolerance(abs(y_pos), bounds=(-0.05, 0.05), margin=0.05, sigmoid='quadratic')
		# straightness = (3 + straightness) / 4
		straightness = self._tolerance(float(ball_pos[1]), bounds=(-0.1, 0.1), margin=0.05, sigmoid='quadratic')
		straightness = (3 + straightness) / 4
		goal_distance,ball_distance,h_ball = self.get_ball_rel_reward(obs)
		com_vel = self.sim.data.joint('root').qvel.copy()
		horizontal_velocity = com_vel[0]

		ball_vel = (ball_pos - prev_ball) / dt
		v_toward_goal = float(ball_vel[0])  
		vel_reward = 1

		move = self._tolerance(horizontal_velocity,
							bounds=(_MOVE_SPEED, float('inf')),
							margin=_MOVE_SPEED, value_at_margin=0,
							sigmoid='linear')
		move = (5*move + 1) / 6
		# if(ball_distance<0.2):
		# 	move = max(1 - (horizontal_velocity/5),0)
		# init_posture_rew = (init_posture_rew + 4)/5
		ball_to_goal = 1
		ball_height = 1
		if(ball_distance<0.2 and stand_reward>=0.75):
			ball_to_goal = self._tolerance(goal_distance,
							margin=5, value_at_margin=0.8,
							sigmoid='quadratic')
			ball_to_goal = (1+ 2*(ball_to_goal))
			if ball_distance<0.15:
				move = max(1 - abs(horizontal_velocity) / 2.0, 0)
				if ball_pos[0]<40.1:
					small_control = 1

			if ball_distance<0.10:
				toe_ball_rew = 1
				vel_reward = self._tolerance(v_toward_goal,
							bounds=(10.0,20.0),
							margin=5, value_at_margin=0.5,
							sigmoid='linear')
				vel_reward = 1 + 3*vel_reward
				ball_height = 1 + 2*self._tolerance(h_ball, bounds=(0.2, 2), 
										margin=0.1, value_at_margin=0.2, sigmoid='linear')
			# slow = max(1 - abs(horizontal_velocity) / 1.0, 0)  
			# move = 0.3 * move + 0.7 * slow   
		# grf_reward = self.foot_grf_rew(obs,speed=horizontal_velocity)
		# grf_reward = (grf_reward + 4)/5
		# ball_h_rew = self._tolerance(h_ball, bounds=(0.1, 0.2),
		# 					margin=0.05, value_at_margin=0.72,
		# 					sigmoid='quadratic')
		# ball_h_rew = (ball_h_rew + 4)/5
		# if h_ball<0.2 and h_ball>0.1:
		# 	ball_h_rew = 10*h_ball
		# if h_ball>0.2:
		# 	ball_h_rew = 1e-3
		rew_components = dict(
			small_control=small_control,
			goal_score = goal_score,
			stand_reward=stand_reward,
			move=move,
			straightness=straightness,
			toe_ball_rew=toe_ball_rew,
			init_posture_rew=init_posture_rew,
			# grf_reward=grf_reward,
			# ball_h_rew=ball_h_rew,
			ball_to_goal=ball_to_goal
		)
		# print(init_posture_rew)
		# pain = self.get_jnt_limit_violation()
		# print()
		# print(quaternion_to_axis_angle(obs['torso_angle']))
		# print(obs['torso_angle'])
		# print("horizontal_velocity: ", horizontal_velocity)
		# print("height:", self.env.sim.data.site('head').xpos[2], "standing:", standing)
		# print(small_control * stand_reward * move, "small_control:", small_control, "stand_reward:", stand_reward, "move:", move)
		# return goal_score * small_control * stand_reward * move * straightness * ball_to_goal * ball_h_rew# * (1 - self._get_fallen_condition())
		# return small_control * stand_reward * move * straightness * init_posture_rew * grf_reward

		self._prev_ball_pos = ball_pos.copy()
		self._prev_time = t
		final_reward = 1
		# for k, v in rew_components.items():
		# 	final_reward *= v
		# final_reward = goal_score * small_control * stand_reward * move * straightness * ball_to_goal * ball_h_rew
		# final_reward = small_control * stand_reward * move * straightness
		final_reward = goal_score * small_control * stand_reward * move * toe_ball_rew * ball_to_goal * vel_reward * straightness * ball_height
		# final_reward = small_control * stand_reward * move * straightness * init_posture_rew 
		return final_reward



	# def _get_torso_posture_reward(self, obs):
	# 	qpos_curr = obs['internal_qpos'][[self.env.myo_joints.index(j) for j in self.torso_joints]]
	# 	qpos_init = self.init_qpos[self.torso_qpos_indices]

	# 	deviation = np.sum((qpos_curr - qpos_init) ** 2)
	# 	reward = np.exp(-5 * deviation)
	# 	return reward


	def get_ball_rel_reward(self,obs):
		ball_pos = self.env.sim.data.body(self.soccer_ball_id).xpos.copy()
		h_ball = ball_pos[2] - 0.1170
		r_toe = np.array(obs['r_toe_pos'])[0:2].copy()
		l_toe = np.array(obs['l_toe_pos'])[0:2].copy()
		# toe-to-ball horizontal distance (min of two toes)
		d_r = np.linalg.norm(r_toe - ball_pos[0:2])
		d_l = np.linalg.norm(l_toe - ball_pos[0:2])
		d_toe = min(d_r, d_l)
		goal_distance = np.abs(ball_pos[0]-self.GOAL_X_POS)
		ball_distance = d_toe
		# print(goal_distance, ball_distance, h_ball )
		if(ball_pos[0]>40.1):
			ball_distance = 0
		return goal_distance, ball_distance, h_ball 

	def get_torso_uprightness(self,obs):
		# get torso orientation quaternion (x, y, z, w)
		torso_quat = obs['torso_angle'].copy()

		# convert quaternion to intrinsic (body frame) Euler angles (roll, pitch, yaw)
		roll, pitch, yaw = quat2euler_intrinsic(torso_quat)

		# pitch is zero when torso is upright
		torso_pitch = pitch

		# smooth uprightness measure: 1 when pitch=0, decreases as torso tilts
		uprightness = np.cos(abs(torso_pitch)) 

		return torso_pitch, uprightness


	def _get_reward_phase(self):
		"""Additive, state-dependent reward that only uses obs_dict fields."""
		obs = self.env.get_obs_dict(self.env.sim)  
		t = float(obs['time'].item()) if hasattr(obs['time'], 'item') else float(obs['time'])
		# bookkeeping for dt and previous ball pos
		prev_t = getattr(self, "_prev_time", t)
		dt = max(1e-6, t - prev_t)
		prev_ball = getattr(self, "_prev_ball_pos", obs['ball_pos'].copy())

		# Basic quantities
		ball_pos = np.array(obs['ball_pos']).copy()        # shape (3,)
		root_qpos = np.array(obs['model_root_pos']).copy() # root_qpos[0:3] -> x,y,z
		root_xy = root_qpos[0:2].copy()
		root_z = float(root_qpos[2])
		r_toe = np.array(obs['r_toe_pos'])[0].copy()
		l_toe = np.array(obs['l_toe_pos'])[0].copy()

		# toe-to-ball horizontal distance (min of two toes)
		d_r = np.linalg.norm(r_toe - ball_pos[0])
		d_l = np.linalg.norm(l_toe - ball_pos[0])
		d_toe = min(d_r, d_l)

		# agent-to-ball center distance (in plane)
		d_root_ball = np.linalg.norm(root_xy - ball_pos[0:2])

		# ---------- Dense shaping components ----------
		# (a) posture: prefer pelvis/root height above a threshold
		z_target = 0.9
		off = (abs(z_target - root_z)/z_target)*100
		r_posture = self._tolerance(root_z, bounds=(z_target, z_target+0.03), margin=0.3)
		print(r_posture,off)
		# (b) alignment: small lateral offset 
		lateral = abs(ball_pos[1] - root_xy[1])
		r_lateral = self._tolerance(lateral, bounds=(0.0, 0.1), margin=0.95, value_at_margin=0.75, sigmoid='quadratic')
		r_align_y = r_lateral


		root_vel = self.sim.data.joint('root').qvel.copy()
		com_x_vel = float(root_vel[0]) # Forward velocity

		# (c) approach: potential-like dense reward based on toe->ball distance (stronger close)
		# use exponential shaping so reward increases smoothly as toe approaches ball
		r_approach = float(np.exp(-1.0 * d_toe))  # ~1 near zero, ~small when far

		# (d) ball forward velocity toward goal (estimated from prev pos)
		ball_vel = (ball_pos - prev_ball) / dt
		v_toward_goal = float(ball_vel[0])  # positive -> toward +x goal
		# only reward positive x-velocity; squash with tanh to make scale robust
		r_ball_vel = float(np.tanh(np.clip(v_toward_goal / 4.0, -10.0, 10.0))) if v_toward_goal > 0.0 else 0.0

		# (e) foot height bonus (optional): reward toe height being appropriate for kick (we can approximate by checking z of toe if you have it)
		r_toe_z = float(obs['r_toe_pos'][2]) if len(obs['r_toe_pos']) >= 3 else None
		r_toe_height = self._tolerance(r_toe_z, bounds=(0.05, 0.30), margin=0.1, value_at_margin=0.2, sigmoid='quadratic')

		# (f) control cost (energy / jerky motions)
		act = np.array(obs['act']).copy() 
		pen_control = float(np.mean(np.square(act))) if act.size > 0 else 0.0

		# neutral_torso = np.array([1, 0, 0, 0])   

		# torso_q = obs["torso_angle"]
		# torso_up = quat_apply(q, np.array([0, 0, 1])) 
		# upright = torso_up[2]    # z-component. 1 = perfectly upright, -1 = upside down
		# torso_stability = np.clip(upright, 0, 1)   # only reward upright, not inverted
		# print(torso_stability)

		r_cyclic = self._get_cyclic_reward(obs)  
		k = 2.0 
		r_cyclic =  np.exp(-k * r_cyclic)

		com_vel = self.sim.data.joint('root').qvel.copy()
		horizontal_velocity = com_vel[0]
		target_fwd_vel = 1.2  # Target velocity in m/s
    
    	# Reward for matching the target forward velocity
		r_forward_vel = np.exp(-2.0 * abs(horizontal_velocity - target_fwd_vel))
    
    	# Gate the velocity reward with the posture reward.
    	# If posture is poor (r_posture is low), this entire term becomes negligible.
		r_stable_walk = r_posture * r_forward_vel
		# ---------- state dependent weights ----------
		# far: emphasize approach; near: emphasize ball velocity (kick) and accuracy
		D_far = 0.9   # meters
		D_kick = 0.20 # toe radius to ball inside which we focus on kicking

		if d_toe > D_far:
			w_stable_walk = 4.0
			w_approach = 2.5
			w_kick = 0
			w_align = 0.3
			w_toe = 0
			w_gait = 0.5

		elif d_toe > D_kick and d_toe < D_far:
			# approaching / positioning region
			w_stable_walk = 2.0
			w_approach = 1.5
			w_kick = 1.5
			w_align = 1.0
			w_toe = 0
			w_gait = 0.5

		else:
			# close: focus on imparting velocity to the ball
			w_stable_walk = 0.5
			w_approach = 0.2
			w_kick = 6.0   
			w_align = 1.0
			w_toe = 1
			w_gait = 0

		w_posture = 4.0
		w_not_fallen = 0.01
		w_act = 0.1
		goal_bonus = 1000.0

		weights = [w_posture, w_align, w_approach, w_kick, w_toe, w_gait]
		norm_factor = sum(weights)

		w_posture  /= norm_factor
		w_align    /= norm_factor
		w_approach /= norm_factor
		w_kick     /= norm_factor
		w_toe      /= norm_factor
		w_gait     /= norm_factor

		dense_sum = (
			w_posture * r_posture +
			w_align * r_align_y +
			w_stable_walk * r_stable_walk +
			w_approach * r_approach +
			w_kick * r_ball_vel +
			w_toe * r_toe_height + 
			w_gait * r_cyclic
		)


		# penalties
		control_pen = w_act * pen_control
		# time_pen = w_time  # per step (we add this constant to encourage finishing earlier)

		# sparse success bonus check
		goal_scored = self._goal_scored_condition()
		sparse_bonus = goal_bonus if goal_scored else 0.0

		fallen = self._get_fallen_condition()
		
		if not fallen and not self.flag:
			stay_up_rew = 0.01
		else:
			self.flag = True
			stay_up_rew = 0
		reward = float(dense_sum + stay_up_rew - control_pen + sparse_bonus)

		self._prev_ball_pos = ball_pos.copy()
		self._prev_time = t

		rew_components = dict(
            posture      = w_posture  * r_posture,
			stable_walk  = w_stable_walk * r_stable_walk,
            align        = w_align    * r_align_y,
            approach     = w_approach * r_approach,
            kick         = w_kick     * r_ball_vel,
            toe_height   = w_toe      * r_toe_height,
            gait_cycle   = w_gait     * r_cyclic,
            stay_up_rew = stay_up_rew,
            control_pen  = -control_pen,
            sparse_bonus = sparse_bonus
        )
		# print(rew_components)
		return reward, rew_components


	def _get_reward(self):
		"""Returns a reward to the agent."""
		_STAND_HEIGHT = self.cfg.stand_height
		_MOVE_SPEED = self.cfg.move_speed
		obs = self.env.get_obs_dict(self.env.sim)
		h = self.env.sim.data.site('head').xpos[2]
		# print(h)

		# NEW
		root_qpos = np.array(obs['model_root_pos']).copy() # root_qpos[0:3] -> x,y,z
		root_xy = root_qpos[0:2].copy()
		root_z = float(root_qpos[2])
		z_target = 0.9
		standing = self._tolerance(root_z, bounds=(z_target, z_target+0.03), margin=0.3)
		# standing = self._tolerance(h,
		# 							bounds=(_STAND_HEIGHT, float('inf')),
		# 							margin=_STAND_HEIGHT/4)
		stand_reward = standing * standing
		act = self.env.sim.data.act[:].copy()
		act_mag = np.mean(np.square(act)) if self.env.sim.model.na !=0 else 0
		small_control = self._tolerance(act_mag, margin=1,
										value_at_margin=0,
										sigmoid='quadratic')
		small_control = (4 + small_control) / 5
		small_control = 1
		if_goal_scored = self._goal_scored_condition() 
		# if_goal_scored = 1
		goal_score = 1		
		if if_goal_scored:
			goal_score = 10
		fallen = self._get_fallen_condition()
		# pain = self.get_jnt_limit_violation() # Joint limit violation torque as pain score
		pos = self.env.sim.data.joint('root').qpos.copy()
		y_pos = pos[1]
		straightness = self._tolerance(abs(y_pos), bounds=(-0.3, 0.3), margin=0.1, sigmoid='quadratic')
		straightness = (3 + straightness) / 4
		goal_distance,ball_distance,h_ball = self.get_ball_rel_reward(obs)
		com_vel = self.sim.data.joint('root').qvel.copy()
		horizontal_velocity = com_vel[0]
		move = self._tolerance(horizontal_velocity,
							bounds=(_MOVE_SPEED, float('inf')),
							margin=_MOVE_SPEED, value_at_margin=0,
							sigmoid='linear')
		move = (5*move + 1) / 6
		if(ball_distance<0.2):
			move = max(1 - (horizontal_velocity/5),0)
		init_posture_rew = self._get_torso_posture_reward(obs)
		init_posture_rew = (init_posture_rew + 4)/5
		ball_to_goal = 3/5
		if(ball_distance<0.1):
			ball_to_goal = self._tolerance(goal_distance,
							margin=5, value_at_margin=0.7,
							sigmoid='quadratic')
			ball_to_goal = (ball_to_goal + 4)/5
		grf_reward = self.foot_grf_rew(obs,speed=horizontal_velocity)
		grf_reward = (grf_reward + 4)/5
		ball_h_rew = self._tolerance(h_ball, bounds=(0.1, 0.2),
							margin=0.05, value_at_margin=0.72,
							sigmoid='quadratic')
		ball_h_rew = (ball_h_rew + 4)/5
		if h_ball<0.2 and h_ball>0.1:
			ball_h_rew = 10*h_ball
		if h_ball>0.2:
			ball_h_rew = 1e-3
		rew_components = dict(
			small_control=small_control,
			goal_score = goal_score,
			stand_reward=stand_reward,
			move=move,
			straightness=straightness,
			# init_posture_rew=init_posture_rew,
			# grf_reward=grf_reward,
			# ball_h_rew=ball_h_rew,
			# ball_to_goal=ball_to_goal
		)
		# print(init_posture_rew)
		# pain = self.get_jnt_limit_violation()
		# print()
		# print(quaternion_to_axis_angle(obs['torso_angle']))
		# print(obs['torso_angle'])
		# print("horizontal_velocity: ", horizontal_velocity)
		# print("height:", self.env.sim.data.site('head').xpos[2], "standing:", standing)
		# print(small_control * stand_reward * move, "small_control:", small_control, "stand_reward:", stand_reward, "move:", move)
		# return goal_score * small_control * stand_reward * move * straightness * ball_to_goal * ball_h_rew# * (1 - self._get_fallen_condition())
		# return small_control * stand_reward * move * straightness * init_posture_rew * grf_reward
		final_reward = 1
		for k, v in rew_components.items():
			final_reward *= v
		# final_reward = goal_score * small_control * stand_reward * move * straightness * ball_to_goal * ball_h_rew
		final_reward =  small_control * stand_reward * move 
		# final_reward = small_control * stand_reward * move * straightness * init_posture_rew 
		return final_reward, rew_components

	def normalize_force(self, f, scale=500.0):
		# smooth 0 to 1 mapping
		return np.tanh(f / scale)

	def foot_grf_rew(self,obs,
						scale=500.0,
						gamma_single_support=0.3,
						speed=0.0,
						max_speed=1.5,
						contact_change=None,
						contact_change_penalty_coef=0.0):
		"""
		obs: dict with obs['grf'] = [r_foot, r_toes, l_foot, l_toes]
		scale: force scale for tanh normalization
		gamma_single_support: weight for single support term in combined reward (0..1)
		speed: current forward speed of agent (scalar). If unknown pass 0.
		max_speed: value to normalize speed to 0..1
		contact_change: optional scalar describing recent change in contacts.
						If provided, a small penalty will be applied to discourage bouncing.
		contact_change_penalty_coef: multiplier for that penalty.
		"""
		r_foot, r_toes, l_foot, l_toes = obs["grf"]
		scale_foot = 25
		scale_toe = 100
		# normalized contact scores in [0,1)
		r_foot_n = self.normalize_force(r_foot, scale_foot)
		r_toes_n = self.normalize_force(r_toes, scale_toe)
		l_foot_n = self.normalize_force(l_foot, scale_foot)
		l_toes_n = self.normalize_force(l_toes, scale_toe)

		# per foot full contact score: product requires both pads to be present
		right_full = r_foot_n * r_toes_n    # near 1 when both are grounded strongly
		left_full  = l_foot_n * l_toes_n

		# double support reward in [0,1]: average of both full foot scores
		double_support = 0.5 * (right_full + left_full)

		# single support reward in [0,1]:
		# reward one foot being strong while the other is weak.
		right_single = right_full * (1.0 - left_full)
		left_single  = left_full  * (1.0 - right_full)
		single_support = max(right_single, left_single)  # which foot is supporting

		# encourage single support only when agent is moving forward
		speed_term = min(max(speed / max_speed, 0.0), 1.0)

		# combine: contact stability + single support times speed
		reward = (1.0 - gamma_single_support) * double_support \
				+ gamma_single_support * single_support * (0.1 + 0.9 * speed_term)

		# clamp to [0,1]
		reward = float(np.clip(reward, 0.0, 1.0))
		return reward


	def _get_fallen_condition(self):
		"""
		Checks if the agent has fallen by checking the if the height of the pelvis is too near to the ground
		"""
		pelvis = self.env.sim.data.body('pelvis').xpos
		if pelvis[2] < 0.2:
			return 1
		else:
			return 0
		
	def _get_grf(self):
		return np.array([self.sim.data.sensor(sens_name).data[0] for sens_name in self.grf_sensor_names]).copy()
	
	# def _get_joint_qpos(self):
	# 	'''
	# 	Return a list of joint qpos from the predefined list of joint names
	# 	'''
	# 	return np.array([self.sim.data.joint(jnt).qpos[0].copy() for jnt in self.torso_joints])

	# def _get_torso_posture_reward(self,obs):
	# 	torso_indices = [self.env.myo_joints.index(j) for j in self.torso_joints]
	# 	# Current joint angles
	# 	qpos_curr = self._get_joint_qpos()
	# 	# Initial joint angles
	# 	qpos_init = np.array([self.sim.model.joint(jnt).qpos0[0] for jnt in self.torso_joints])
	# 	# squared deviation
	# 	deviation = np.sum((qpos_curr - qpos_init) ** 2)
	# 	reward = np.exp(-(5) * deviation)  # scaling factor = 1/5
	# 	return reward

	def _get_torso_posture_reward(self, obs):
		qpos_curr = obs['internal_qpos'][[self.env.myo_joints.index(j) for j in self.torso_joints]]
		qpos_init = self.init_qpos[self.torso_qpos_indices]

		deviation = np.sum((qpos_curr - qpos_init) ** 2)
		return deviation


	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.sim.renderer.render_offscreen(
			width=1920, height=1080, camera_id=self.camera_id
		).copy()
	
	def _goal_scored_condition(self):
		"""
		Checks if the ball has entered the goal.
		The ball must cross GOAL_X_POS and be within the GOAL_Y/Z bounds.
		"""
		ball_pos = self.env.sim.data.body(self.soccer_ball_id).xpos.copy()
		is_x_past_goal = ball_pos[0] >= self.GOAL_X_POS
		is_y_in_bounds = self.GOAL_Y_MIN <= ball_pos[1] <= self.GOAL_Y_MAX
		is_z_in_bounds = self.GOAL_Z_MIN <= ball_pos[2] <= self.GOAL_Z_MAX
		return bool(is_x_past_goal and is_y_in_bounds and is_z_in_bounds)

	def _tolerance(self, x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
              value_at_margin=0.1):
		"""Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.
		Adapted from DMcontrol.

		Args:
			x: A scalar or numpy array.
			bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
			the target interval. These can be infinite if the interval is unbounded
			at one or both ends, or they can be equal to one another if the target
			value is exact.
			margin: Float. Parameter that controls how steeply the output decreases as
			`x` moves out-of-bounds.
			* If `margin == 0` then the output will be 0 for all values of `x`
				outside of `bounds`.
			* If `margin > 0` then the output will decrease sigmoidally with
				increasing distance from the nearest bound.
			sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
			'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
			value_at_margin: A float between 0 and 1 specifying the output value when
			the distance from `x` to the nearest bound is equal to `margin`. Ignored
			if `margin == 0`.

		Returns:
			A float or numpy array with values between 0.0 and 1.0.

		Raises:
			ValueError: If `bounds[0] > bounds[1]`.
			ValueError: If `margin` is negative.
		"""
		lower, upper = bounds
		if lower > upper:
			raise ValueError('Lower bound must be <= upper bound.')
		if margin < 0:
			raise ValueError('`margin` must be non-negative.')

		in_bounds = np.logical_and(lower <= x, x <= upper)
		if margin == 0:
			value = np.where(in_bounds, 1.0, 0.0)
		else:
			d = np.where(x < lower, lower - x, x - upper) / margin
			value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

		return float(value) if np.isscalar(x) else value

def _sigmoids(x, value_at_1, sigmoid):
	"""Returns 1 when `x` == 0, between 0 and 1 otherwise.

	Args:
		x: A scalar or numpy array.
		value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
		sigmoid: String, choice of sigmoid type.

	Returns:
		A numpy array with values between 0.0 and 1.0.

	Raises:
		ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
		`quadratic` sigmoids which allow `value_at_1` == 0.
		ValueError: If `sigmoid` is of an unknown type.
	"""
	if sigmoid in ('cosine', 'linear', 'quadratic'):
		if not 0 <= value_at_1 < 1:
			raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
						'got {}.'.format(value_at_1))
	else:
		if not 0 < value_at_1 < 1:
			raise ValueError('`value_at_1` must be strictly between 0 and 1, '
						'got {}.'.format(value_at_1))

	if sigmoid == 'gaussian':
		scale = np.sqrt(-2 * np.log(value_at_1))
		return np.exp(-0.5 * (x*scale)**2)

	elif sigmoid == 'hyperbolic':
		scale = np.arccosh(1/value_at_1)
		return 1 / np.cosh(x*scale)

	elif sigmoid == 'long_tail':
		scale = np.sqrt(1/value_at_1 - 1)
		return 1 / ((x*scale)**2 + 1)

	elif sigmoid == 'reciprocal':
		scale = 1/value_at_1 - 1
		return 1 / (abs(x)*scale + 1)

	elif sigmoid == 'cosine':
		scale = np.arccos(2*value_at_1 - 1) / np.pi
		scaled_x = x*scale
		with warnings.catch_warnings():
			warnings.filterwarnings(
				action='ignore', message='invalid value encountered in cos')
			cos_pi_scaled_x = np.cos(np.pi*scaled_x)
		return np.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x)/2, 0.0)

	elif sigmoid == 'linear':
		scale = 1-value_at_1
		scaled_x = x*scale
		return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

	elif sigmoid == 'quadratic':
		scale = np.sqrt(1-value_at_1)
		scaled_x = x*scale
		return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

	elif sigmoid == 'tanh_squared':
		scale = np.arctanh(np.sqrt(1-value_at_1))
		return 1 - np.tanh(x*scale)**2

	else:
		raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def make_env(cfg):
	"""
	Make Myosuite environment.
	"""
	if not cfg.task in MYOSUITE_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	import myosuite
	from myosuite.utils import gym as gym_utils
	env = gym_utils.make(MYOSUITE_TASKS[cfg.task])
	env = MyoSuiteWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=200)
	env = ActionSpaceWrapper(env)
	# env.max_episode_steps = 200
	return env
