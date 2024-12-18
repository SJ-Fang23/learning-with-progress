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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="ppo_experiment")
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--env_name', type=str, default="NutAssemblySquare")

    args = parser.parse_args()
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.env_name == "NutAssemblySquare":
        dataset_path = os.path.join(project_path,"human-demo/square/low_dim_v141.hdf5")
    elif args.env_name == "PickPlaceCan":
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

    # env = suite.make(
    #     "PickPlaceCanModified",
    #     **make_env_kwargs,
    # )
    envs = make_vec_env_robosuite(
        args.env_name,
        obs_keys = ["object-state","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        rng=np.random.default_rng(SEED),
        n_envs=12,
        parallel=True,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
        env_make_kwargs=make_env_kwargs,
    )

    learner = PPO(
        env=envs,
        policy=MlpPolicy,
        batch_size=256,
        ent_coef=0.00,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
        vf_coef=0.5,
        n_epochs=10,
        seed=SEED,
    )
    obs_keys = ["object-state", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    traning_time = 1000
    training_round = 2000
    start = 0
    if args.checkpoint != "":
        learner = PPO.load(f"{project_path}/checkpoints/{args.exp_name}/"+args.checkpoint+"/gen_policy/model", env=envs)
        training_round += int(args.checkpoint)
        start = int(args.checkpoint) + 1

    reward_net_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner_rewards_before_training = evaluate_policy(learner, envs, n_eval_episodes=10)
    print("mean reward before training:", np.mean(learner_rewards_before_training))
    record_file = os.path.join(project_path, "ppo_learning_test", args.exp_name + ".txt")
    with open(record_file, "w") as f:
        f.write("mean reward before training:" + str(np.mean(learner_rewards_before_training)) + "\n")
    f.close()
    for i in range(start, training_round):
        
        learner.learn(total_timesteps=traning_time)
        if i % 20 == 0:
            learner.save(f"{project_path}/checkpoints/{args.exp_name}/"+str(i)+"/gen_policy/model")
            learner_rewards = evaluate_policy(learner, envs, n_eval_episodes=10)
            print("mean reward at round", i, ":", np.mean(learner_rewards))
        with open(record_file, "a") as f:
            f.write("mean reward at round " + str(i) + ":" + str(np.mean(learner_rewards)) + "\n")
        f.close()
        # if np.mean(learner_rewards) > 100:
        #     break