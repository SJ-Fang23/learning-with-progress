from robosuite.wrappers import Wrapper
from collections import deque
import numpy as np

class SequentialObservationWrapper(Wrapper):
    def __init__(self,
                  env,
                sequential_observation_keys = ["robot0_gripper_qpos"], 
                sequential_observation_length = 50,
                use_half_gripper_obs = True):
        super().__init__(env)
        self.sequential_observation_keys = sequential_observation_keys
        self.sequential_observation_length = sequential_observation_length
        self.obs_queue = deque(maxlen=self.sequential_observation_length)
        self.use_half_gripper_obs = use_half_gripper_obs

    def reset(self):
        obs = self.env.reset()

        # get the observations to be stacked
        obs_to_be_sequenced = {key: obs[key] for key in self.sequential_observation_keys}

        # if use_half_gripper_obs is True, then we only use the first half of the gripper observation
        if self.use_half_gripper_obs:
            obs_to_be_sequenced["robot0_gripper_qpos"] = obs_to_be_sequenced["robot0_gripper_qpos"][:1]

        # fill the queue with the same observation
        for _ in range(self.sequential_observation_length):
            self.obs_queue.append(obs_to_be_sequenced)
        
        # replace the corresponding key in the observation dict
        obs_to_be_replaced = {key: np.concatenate([obs[key] for obs in self.obs_queue], axis=0) for key in self.sequential_observation_keys}
        # update the observation
        obs.update(obs_to_be_replaced)
        return obs
    
    def step(self, action):
        
        obs, reward, done, info = self.env.step(action)
        # push the corresponding observation to the queue
        obs_to_be_sequenced = {key: obs[key] for key in self.sequential_observation_keys}

        # if use_half_gripper_obs is True, then we only use the first half of the gripper observation
        if self.use_half_gripper_obs:
            obs_to_be_sequenced["robot0_gripper_qpos"] = obs_to_be_sequenced["robot0_gripper_qpos"][:1]

        self.obs_queue.append(obs_to_be_sequenced)

        # get the observation to be replaced
        obs_to_be_replaced = {key: np.concatenate([obs[key] for obs in self.obs_queue], axis=0) for key in self.sequential_observation_keys}
        # update the observation
        obs.update(obs_to_be_replaced)
        return obs, reward, done, info
        