# collect demostrations and annotations from policy rollouts
import h5py
import os
import json
import robosuite as suite
import sys
import time
from envs.pickplace import*
import numpy as np
from robosuite.controllers import load_controller_config
import argparse
from utils.demostration_utils import load_data_to_h5py
from utils.annotation_utils import write_to_json
import cv2
from envs.wrappers import SequentialObservationWrapper
from envs.pickplace import PickPlaceCan
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from annotation.replay_collector import replay_trajectory_and_collect_progress


def collect_policy_rollouts(
        policy,
        env_name, 
        exp_name,
        obs_seq_len,
        sequence_keys,
        env_kwargs,
        num_rollouts,
        max_rollout_length,
        online_data_path,
        render=False,
):
    
    SEED = 2024

    if render:
        env_kwargs["has_renderer"] = True
        env_kwargs["has_offscreen_renderer"] = True

    env = suite.make(env_name, **env_kwargs)
    if  obs_seq_len > 1:
        env = SequentialObservationWrapper(env, 
                                           sequential_observation_keys = sequence_keys, 
                                           sequential_observation_length = obs_seq_len, 
                                           use_half_gripper_obs = True)
    
    env = GymWrapper(env, keys = ["object-state", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"])
    
    rollout_data = dict(
        data = dict(),
    )
    
    for i in range(num_rollouts):
        single_rollout = dict(
            obs = [],
            actions = [],
            dones = [],
            rewards = [],
            states = [],
        )

        obs = env.reset()
        # make gym style observation to stable baselines style observation
        obs = np.array(obs[0])
        print("obs shape", obs.shape)
        print(obs)
        done = False

        # initial state
        single_rollout["states"].append(env.sim.get_state().flatten())

        for j in range(max_rollout_length):
            action, _ = policy.predict(obs.reshape(1,-1))
            action = action.squeeze()
            single_rollout["obs"].append(obs)
            single_rollout["actions"].append(action)
            single_rollout["dones"].append(done)
            
            obs, reward, done, _, info = env.step(action)

            if render:
                env.render()
            
            single_rollout["rewards"].append(reward)

            if done:
                break
        
        rollout_data["data"][f'rollout_{i}'] = single_rollout

    # add env kwargs to attributes
    env_args = dict(
        env_name = env_name,
        env_kwargs = env_kwargs,
    )
    env_args = dict()

    print("env_args", env_args)
    # print all the types of env_args
    print("env_name", type(env_name))
    for key, value in env_kwargs.items():
        print(key, type(value))

    for key, value in env_kwargs["controller_configs"].items():
        print(key, type(value))

    # save the data
    if not os.path.exists(os.path.dirname(online_data_path)):
        os.makedirs(os.path.dirname(online_data_path))
    f = h5py.File(online_data_path, 'w')
    for key, value in rollout_data.items():
        group = f.create_group(key)
        if key == "data":
            group.attrs["env_args"] = np._strinenv_args
            group.attrs["obs_seq_len"] = obs_seq_len 
            group.attrs["sequence_keys"] = sequence_keys
            group.attrs["exp_name"] = exp_name
        for key2, value2 in value.items():
            group.create_dataset(key2, data=value2)

    f.close()
    env.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_rollouts', type=int, default=10)
    parser.add_argument('--max_rollout_length', type=int, default=250)
    parser.add_argument('--exp_name', type=str, default="default_experiment")
    parser.add_argument('--checkpoint', type=str, default="300")
    parser.add_argument('-l', '--obs_seq_len', type=int, default=1)
    parser.add_argument('-k', "--sequence_keys", nargs='+', default=["robot0_gripper_qpos"])
    parser.add_argument('-r', '--render', type=bool, default=False)

    parser.add_argument('--online_data_path', type=str, default=None)

    args = parser.parse_args()

    if not args.online_data_path:
        online_data_path = f"online_data_{parser.parse_args().exp_name}_{parser.parse_args().checkpoint}.hdf5"
    else:
        online_data_path = args.online_data_path


    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,"human-demo/can-pick/low_dim_v141.hdf5")
    f= h5py.File(dataset_path,'r')

    # read the environment meta data
    env_kwargs = json.loads(f["data"].attrs["env_args"])["env_kwargs"]

    env = suite.make(
        "PickPlaceCan",
        **env_kwargs,
    )

    # sequential observation wrapper if needed
    
    
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    policy = PPO.load(f"{project_path}/checkpoints/{args.exp_name}/{args.checkpoint}/gen_policy/model")
    policy.policy.eval()

    online_data_path = os.path.join(project_path, "online_data",online_data_path)

    collect_policy_rollouts(
        policy = policy,
        env_name = "PickPlaceCan",
        env_kwargs = env_kwargs,
        num_rollouts = args.num_rollouts,
        max_rollout_length = args.max_rollout_length,
        online_data_path = online_data_path,
        render = args.render,
        obs_seq_len = args.obs_seq_len,
        sequence_keys = args.sequence_keys,
        exp_name=args.exp_name,
    )

    # collect the progress data
    replay_trajectory_and_collect_progress(
        online_data_path,
        reply_demo_indicies = None,
        replay_demo_nums= None,
        collect_progress_times= 10,
        env_kwargs= env_kwargs,
    )



        

