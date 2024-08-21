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

    # print("env_args", env_args)
    # print all the types of env_args
    # print("env_name", type(env_name))
    for key, value in env_kwargs.items():
        print(key, type(value))

    for key, value in env_kwargs["controller_configs"].items():
        print(key, type(value))

    # save the data
    if not os.path.exists(os.path.dirname(online_data_path)):
        os.makedirs(os.path.dirname(online_data_path))
    
    # save the data to npz file
    np.savez(online_data_path, **rollout_data)

    f.close()
    env.close()


def replay_online_trajectory_and_collect_progress(
        env_name, 
        env_kwargs,
        online_data_path,
        progress_data_folder = "online_progress_data",
        collect_progress_times = 10,
):
    env_kwargs["has_renderer"] = True
    env_kwargs["reward_shaping"] = True

    env: PickPlaceCan = suite.make(env_name, **env_kwargs)

    # load online data
    online_data = np.load(online_data_path, allow_pickle=True)["data"]
    print(type(online_data))
    # to dict
    online_data = online_data.item()

    progress_data = dict()

    # do the replay
    for data in online_data.keys():

        data_values = online_data[data]

        obs = data_values["obs"]
        actions = data_values["actions"]
        dones = data_values["dones"]
        rewards = data_values["rewards"]
        states = data_values["states"]

        env.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        pause_indices = np.linspace(0, len(actions), collect_progress_times+2, dtype=int)[1:-1]
        pause_indices = np.append(pause_indices, len(actions)-1)
        progress = [0]

        print(f"Replaying trajectory {data}, length {len(actions)}")
        for i in range(len(obs)):
            action = actions[i]
            obs, reward, done, info = env.step(action)
            env.render()

            if i == 0:
                time.sleep(1)
            if i in pause_indices:
                progress_data[data] = progress_data.get(data, [])
                # get user input
                user_input = input("Please input the progress data: ")
                # user input must be a float, otherwise ask user to input again
                while not user_input.replace(".", "").isdigit():
                    user_input = input("Please input the progress data: ")
                progress.append(float(user_input))
                
                single_data = dict(
                    start_step = int(pause_indices[np.where(pause_indices == i)[0][0]-1] if np.where(pause_indices == i)[0][0]-1 >= 0 else 0),
                    end_step = i,
                    start_progress = progress[len(progress)-2] if len(progress) >= 2 else 0,
                    end_progress = float(user_input)
                )

                progress_data[data].append(single_data)
                # render the environment
                env.render()
            if i == len(actions) - 1:
                break
        
        write_to_json(progress_data[data], "{}.json".format(data), data_folder=progress_data_folder)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--collect_new_data', action='store_true')
    parser.add_argument('-n', '--num_rollouts', type=int, default=10)
    parser.add_argument('--max_rollout_length', type=int, default=250)
    parser.add_argument('--exp_name', type=str, default="default_experiment")
    parser.add_argument('--checkpoint', type=str, default="300")
    parser.add_argument('-l', '--obs_seq_len', type=int, default=1)
    parser.add_argument('-k', "--sequence_keys", nargs='+', default=["robot0_gripper_qpos"])
    parser.add_argument('-r', '--render', type=bool, default=False)

    parser.add_argument('--online_data_path', type=str, default=None)
    parser.add_argument('--progress_data_folder', type=str, default="online_progress_data")
    parser.add_argument('--collect_progress_times', type=int, default=10)

    args = parser.parse_args()

    if not args.online_data_path:
        online_data_path = f"online_data_{parser.parse_args().exp_name}_{parser.parse_args().checkpoint}.npz"
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

    print(args.collect_new_data)
    if args.collect_new_data:
        print("Collecting new data")
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
    else:
        # check if the online data exists
        if not os.path.exists(online_data_path):
            print("Online data does not exist, please set collect_new_data to True")
            sys.exit(1)
    


    # collect the progress data
    replay_online_trajectory_and_collect_progress(
        env_name = "PickPlaceCan",
        env_kwargs = env_kwargs,
        online_data_path = online_data_path,
        progress_data_folder = args.progress_data_folder,
        collect_progress_times = args.collect_progress_times,
    )
    



        

