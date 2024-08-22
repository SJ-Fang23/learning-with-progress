# evaluate trained model

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

# from imitation.algorithms.adversarial.airl import AIRL
from IRL_lib_mod.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from utils.irl_utils import make_vec_env_robosuite
from utils.demostration_utils import load_dataset_to_trajectories
import os
import h5py
import json
from robosuite.controllers import load_controller_config
from utils.demostration_utils import load_dataset_and_annotations_simutanously
from utils.annotation_utils import read_all_json
from imitation.util import logger as imit_logger
import imitation.scripts.train_adversarial as train_adversarial
import argparse
import robosuite as suite
import torch
from robosuite.wrappers import GymWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="default_experiment")
    parser.add_argument('--checkpoint', type=str, default="260")

    args = parser.parse_args()

    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,"human-demo/can-pick/low_dim_v141.hdf5")
    
    f= h5py.File(dataset_path,'r')
    env_meta = json.loads(f["data"].attrs["env_args"])

    make_env_kwargs = dict(
        robots="Panda",             # load a Sawyer robot and a Panda robot
        gripper_types="default",                # use default grippers per robot arm
        controller_configs=env_meta["env_kwargs"]["controller_configs"],   # each arm is controlled using OSC
        has_renderer=True,                      # on-screen rendering
        render_camera="frontview",              # visualize the "frontview" camera
        has_offscreen_renderer=True,           # no off-screen rendering
        control_freq=20,                        # 20 hz control for applied actions
        horizon=500,                            # each episode terminates after 200 steps
        use_object_obs=True,                   # no observations needed
        use_camera_obs=False,
        reward_shaping=True,
    )

    SEED = 1
#     env = make_vec_env_robosuite(
#     "PickPlaceCan",
#     obs_keys = ["object-state","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
#     rng=np.random.default_rng(SEED),
#     n_envs=1,
#     parallel=True,
#     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
#     env_make_kwargs=make_env_kwargs,
# )

    env = suite.make(
        "PickPlaceCanModified",
        **make_env_kwargs,
    )
    env = GymWrapper(env, keys = ["object-state", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"])
    policy = PPO.load(f"{project_path}/checkpoints/{args.exp_name}/{args.checkpoint}/gen_policy/model", env=env)
    reward_net = (torch.load(f"{project_path}/checkpoints/disc_128/300/reward_train.pt"))
    reward_net.eval()
    reward_net_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_net.to(reward_net_device)
    # estimate_reward = evaluate_policy(policy, env, n_eval_episodes=10)
    # estimate_reward = np.mean(estimate_reward)
    # print(f"Estimated Reward: {estimate_reward}")
    evaluate_times = 10
    obs_keys = ["object-state", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    
    for i in range(evaluate_times):
        obs = env.reset()
        obs = obs[0]
        past_action = np.zeros(7)
        done = False
        cnt = 0
        rewards= 0
        disc_rewards = 0
        while not done:
            #print("I reached here")
            cnt += 1
            # obs = torch.tensor(obs).float().squeeze(0).to(reward_net_device)
            # obs_cpu = obs.cpu().detach().numpy()    
            action, _states, = policy.predict(obs, deterministic=True)
            # print(action)
            # if cnt > 150:
            #     action[6] = 1


            next_obs, reward, next_done,_, info = env.step(action)
            rewards +=reward
            if next_done:
                print("final step reward", reward)
            #next_obs = [next_obs[key] for key in obs_keys]
            # next_obs = next_obs[0]
            # print("next_obs", next_obs) 
            # # print(next_obs)
            # obs_tensor = obs.unsqueeze(0).to(reward_net_device).detach()
            # action_tensor = torch.tensor(action).float().unsqueeze(0).to(reward_net_device)
            # next_obs_tensor = torch.tensor(next_obs).float().unsqueeze(0).to(reward_net_device)
            # done = torch.tensor([0]).float().unsqueeze(0).to(reward_net_device)
            # get the reward from the reward network
            # disc_rew = reward_net(obs_tensor, action_tensor, next_obs_tensor, done)
            # disc_rewards += disc_rew.item()
            obs = next_obs
            past_action = action

            env.render()
            if next_done:
                break
        print(f"Total Reward: {rewards}")
        print(f"Discriminator Reward: {disc_rewards}")