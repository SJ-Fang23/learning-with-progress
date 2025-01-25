# evaluate trained model

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
import scipy
from sklearn.preprocessing import MinMaxScaler
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
import imageio 

class CustomLoggingPolicy(MlpPolicy):
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        global print_cnt
        print_cnt += 1

            # Get the action, value, and log probability from the parent class
        actions, values, log_probs = super().forward(obs, deterministic)
        if print_cnt % 2000 == 0:
            print(f"Actions: {actions[-1].detach().cpu().numpy()}")
                        # Convert actions to NumPy for easier processing
            actions_np = actions.detach().cpu().numpy()
            # Update total actions and count of positive last elements

            positive_last = np.sum(actions_np[:, -1] > 0)
            ratio = positive_last / actions_np.shape[0]
            print(f"Positive ratio: {ratio}")

        # Log the actions (you can adjust the logging as needed)


        # Return the outputs as usual
        return actions, values, log_probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="default_experiment")
    parser.add_argument('--checkpoint', type=str, default="260")
    parser.add_argument('--env_name', type=str, default="Lift")
    parser.add_argument('--dataset_type', type=str, default = "mh")  
    parser.add_argument('--render', type=str, default="on")
    parser.add_argument('--eval_times', type=int, default=10)

    scaler = MinMaxScaler()
    args = parser.parse_args()
    if args.render == "on":
        args.render = True
    else:
        args.render = False
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #dataset_path = os.path.join(project_path,"human-demo/" + args.env_name + "/low_dim_v141_" + args.env_name + "_" + args.dataset_type + ".hdf5")
    # dataset_path = os.path.join(project_path,"human-demo/square/low_dim_v141.hdf5")
    dataset_path = os.path.join(project_path,"learning-with-progress/human-demo/lift/low_dim_v141_lift_ph.hdf5")

    f= h5py.File(dataset_path,'r')
    env_meta = json.loads(f["data"].attrs["env_args"])
    # make_env_kwargs = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
    # # enable rendering
    # make_env_kwargs["has_renderer"] = True
    # make_env_kwargs["reward_shaping"] = True
    # make_env_kwargs['horizon'] = 300
    # print('dataset make_env_kwargs')
    # print(make_env_kwargs)

    make_env_kwargs = dict(
        robots="Panda",             # load a Sawyer robot and a Panda robot
        gripper_types="default",                # use default grippers per robot arm
        controller_configs=env_meta["env_kwargs"]["controller_configs"],   # each arm is controlled using OSC
        has_renderer=  args.render,           # no on-screen renderer
        render_camera="frontview",              # visualize the "frontview" camera
        has_offscreen_renderer=True,           # no off-screen rendering
        control_freq=10,                        # 20 hz control for applied actions
        horizon=300,                            # each episode terminates after 200 steps
        use_object_obs=True,                   # no observations needed
        use_camera_obs=False,
        reward_shaping=True,
    )
    # print("we set the make_env_kwargs")
    # print("make_env_kwargs", make_env_kwargs)

    SEED = 1
    print("chechpoint", args.checkpoint)
    print(args.render)

    env = suite.make(
        args.env_name,
        **make_env_kwargs,
    )
    custom_objects = {
        'policy_class': CustomLoggingPolicy
    }
    reward_net_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy = PPO.load(f"{project_path}/learning-with-progress/checkpoints/{args.exp_name}/{args.checkpoint}/gen_policy/model", custom_objects=custom_objects, device = reward_net_device)
    reward_net = torch.load(f"{project_path}/learning-with-progress/checkpoints/{args.exp_name}/{args.checkpoint}/reward_train.pt", map_location=reward_net_device)
    reward_net.eval()
    reward_net.to(reward_net_device)
    reward_net_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_net.to(reward_net_device)
    video_dir = os.path.join(project_path, "videos", args.exp_name)
    os.makedirs(video_dir, exist_ok=True)
    evaluate_times = args.eval_times
    obs_keys = ["cube_pos", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    
    env_rewards = []
    correlations = []
    normalized_pearson_correlations = []
    normalized_spearman_correlations = []
    success_cnt = 0
    for i in range(evaluate_times):
        obs = env.reset()
        obs = [obs[key] for key in obs_keys]
        obs = np.concatenate(obs)
        past_action = np.zeros(7)
        done = False
        cnt = 0
        rewards= []
        total_disc_rew = []
        frames = []
        while not done:
            
            #action, _states = policy.predict(obs)
            action, _ = policy.predict(obs, deterministic=True)
            cnt += 1
            if args.render:
                env.render()
                frame = env.render()
                frames.append(frame)
            obs = torch.tensor(obs).float().unsqueeze(0).to(reward_net_device)
            obs = obs.cpu().detach().numpy()
            # print("obs", obs)   
            
            #action, _ = policy.predict(obs, deterministic=True)
            action = action.squeeze()
            #print(action)
            # if cnt > 200:
            #     action[6] = 1
            #action = action.cpu().detach().numpy().squeeze()

            next_obs, reward, next_done, info = env.step(action)
            
            next_obs = [next_obs[key] for key in obs_keys]
            next_obs = np.concatenate(next_obs)
            # # print(next_obs)
            obs = torch.tensor(obs).float().unsqueeze(0).to(reward_net_device)
            obs_tensor = obs.unsqueeze(0).to(reward_net_device).detach()
            action_tensor = torch.tensor(action).float().unsqueeze(0).to(reward_net_device)
            next_obs_tensor = torch.tensor(next_obs).float().unsqueeze(0).to(reward_net_device)
            done = torch.tensor([0]).float().unsqueeze(0).to(reward_net_device)
            # get the reward from the reward network
            disc_rew = reward_net(obs_tensor, action_tensor, next_obs_tensor, done)
            total_disc_rew.append(disc_rew.item())
            rewards.append(reward)
            # print(type(reward))
            # print(type(disc_rew.item()))
            obs = next_obs
            past_action = action
            #print(f"Discriminator Reward: {disc_rew}")
            # if action[6] > 0:
            #     print(f"gripper action: {action[6]}")
            if args.render:
                env.render()

                #print("******************Success*********************")
            # print("done", next_done)
            # print("info", info)
            #env.render()
            if next_done:
                print("yessssssss")
                if obs[2] > 0.8565:
                    success_cnt += 1
                break
       # video_path = os.path.join(video_dir, f"episode_{i+1}.mp4")
        
        print(f"Total Discriminator Reward: {sum(total_disc_rew)}")
        print(f"Total Reward: {sum(rewards)}")


        # Normalize rewards and total_disc_rew
        rewards_normalized = scaler.fit_transform(np.array(rewards).reshape(-1, 1)).flatten()
        total_disc_rew_normalized = scaler.fit_transform(np.array(total_disc_rew).reshape(-1, 1)).flatten()
        
        # Compute correlations
        correlation = scipy.stats.spearmanr(rewards, total_disc_rew)
        # normalized_pearson = scipy.stats.pearsonr(rewards_normalized, total_disc_rew_normalized)
        # normalized_spearman = scipy.stats.spearmanr(rewards_normalized, total_disc_rew_normalized)
        
        print(f"Correlation (Spearman): {correlation[0]}")
        # print(f"Normalized Pearson Correlation: {normalized_pearson[0]}")
        # print(f"Normalized Spearman Correlation: {normalized_spearman[0]}")
        
        correlations.append(correlation[0])
        # normalized_pearson_correlations.append(normalized_pearson[0])
        # normalized_spearman_correlations.append(normalized_spearman[0])

        env_rewards.append(sum(rewards))

        # imageio.mimwrite(video_path, frames, fps=20, codec='libx264')
        # print(f"Saved video for episode {i+1} at {video_path}")

    print(f"Success Rate: {success_cnt}/{evaluate_times}")
    print(env_rewards)
    print(correlations)
    print(f"Average Reward: {np.mean(env_rewards)}")
    print(f"Average Correlation: {np.mean(correlations)}")
    print(f"reward list: {env_rewards}")
    # print(f"Average Normalized Pearson Correlation: {np.mean(normalized_pearson_correlations)}")
    # print(f"Average Normalized Spearman Correlation: {np.mean(normalized_spearman_correlations)}")

    #write the results to a file
    results_path = os.path.join(project_path, "results", args.exp_name, args.checkpoint, ".txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        f.write(f"Success Rate: {success_cnt}/{evaluate_times}\n")
        f.write(f"Average Reward: {np.mean(env_rewards)}\n")
        f.write(f"Average Correlation: {np.mean(correlations)}\n")
        f.write(f"reward list: {env_rewards}\n")
        f.write(f"correlation list: {correlations}\n")
        # f.write(f"Average Normalized Pearson Correlation: {np.mean(normalized_pearson_correlations)}\n")
        # f.write(f"Average Normalized Spearman Correlation
