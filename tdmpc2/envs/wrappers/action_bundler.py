import gymnasium as gym
import numpy as np

class ActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.syn_action_shape = 24 + 80  # 24 reduced + 80 direct mappings
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.syn_action_shape,), dtype=np.float32)

        # Define the mapping from reduced to original action space for the first 210 muscles
        self.action_mapping = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], #psoas major right
            1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  #psoas major left
            2: [22], # RA, right
            3: [23], #RA left
            4: [24, 25, 26, 27], #ILpL right
            5: [28, 29, 30, 31], #ILpL left
            6: [32, 33, 34, 35, 36, 37, 38, 39],  #ILpT right
            7: [40, 41, 42, 43, 44, 45, 46, 47], #ILpT left
            8: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68], #LTpT right
            9: [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89], #LTpT left
            10: [90, 91, 92, 93, 94], #LTpL right
            11: [95, 96, 97, 98, 99], #LTpL left
            12: [100, 101, 102, 103, 104, 105, 106], #QL_post right
            13: [107, 108, 109, 110, 111, 112, 113],  #QL_post left
            14: [114, 115, 116, 117, 118],  #QL_mid right
            15: [119, 120, 121, 122, 123],  #QL_mid left
            16: [124, 125, 126, 127, 128, 129 ], #QL_ant right
            17: [130, 131, 132, 133, 134, 135], #QL_ant left
            18: [136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160], #MF right
            19: [161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185], #MF left
            20: [186, 187, 188, 189, 190, 191], #EO right
            21: [192, 193, 194, 195, 196, 197], #IO right
            22: [198, 199, 200, 201, 202, 203], #EO left
            23: [204, 205, 206, 207, 208, 209] #IO left
        }

        # Add the direct mapping for the next 80 muscles (210 to 289)
        # for i in range(24, 89):
            # self.action_mapping[i] = [i + 169]  # Mapping 210 to 290 (offset by 184)
        self.torso_muscles = 210
        self.num_leg_muscles = 80
        for i in range(self.num_leg_muscles):
            reduced_action_index = i + 24
            full_action_index = i + self.torso_muscles
            self.action_mapping[reduced_action_index] = [full_action_index]

    def action(self, action):
        # Map the reduced action space to the full action vector
        assert len(action) == self.syn_action_shape
        # print("action: ",action[:3],action.shape)
        full_action = np.zeros(self.env.action_space.shape)
        for i, indices in self.action_mapping.items():
            full_action[indices] = action[i]
        # print("full action: ",full_action[:30],full_action.shape)
        return full_action
	
    def reset(self, **kwargs):
        self._t = 0
        return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)