# train on ph using binary gripper with manual close gripper chance, and modified hyperparameters


import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from RL_wrapper.ppo_wrapper import ActorCriticPolicyWrapperManualCloseGripper

# from imitation.algorithms.adversarial.airl import AIRL
from IRL_lib_mod.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from envs.wrappers import SequentialObservationWrapper
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
import torch

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="default_experiment")
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default="320")
    parser.add_argument('--load_exp_name', type=str, default="mh_sign_scale_loss_8m_1")
    parser.add_argument('-s', '--sequence_keys', nargs='+', default=[])
    parser.add_argument('-l', '--obs_seq_len', type=int, default=1)


    
    args = parser.parse_args()
    print("sequence keys", args.sequence_keys)
    print("obs_seq_len", args.obs_seq_len)


    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,"human-demo/can-pick/low_dim_v141.hdf5")
    log_dir = os.path.join(project_path,f"logs/{args.exp_name}")
    print(dataset_path)
    f = h5py.File(dataset_path,'r')

    config_path = os.path.join(project_path,"configs/osc_position.json")
    with open(config_path, 'r') as cfg_file:
        configs = json.load(cfg_file)

    controller_config = load_controller_config(default_controller="OSC_POSE")
    env_meta = json.loads(f["data"].attrs["env_args"])
    SEED = 42
    make_env_kwargs = dict(
        robots="Panda",             # load a Sawyer robot and a Panda robot
        gripper_types="default",                # use default grippers per robot arm
        controller_configs=env_meta["env_kwargs"]["controller_configs"],   # each arm is controlled using OSC
        has_renderer=False,                      # on-screen rendering
        render_camera="frontview",              # visualize the "frontview" camera
        has_offscreen_renderer=False,           # no off-screen rendering
        control_freq=20,                        # 20 hz control for applied actions
        horizon=1000,                            # each episode terminates after 300 steps
        use_object_obs=True,                   # no observations needed
        use_camera_obs=False,
        reward_shaping=True,
        
    )

    print("sequence keys", args.sequence_keys)
    if len(args.sequence_keys) > 0:
        sequential_wrapper_kwargs = dict(
            sequential_observation_keys = args.sequence_keys, 
            sequential_observation_length = args.obs_seq_len, 
            use_half_gripper_obs = True
        )

        seqential_wrapper_cls = SequentialObservationWrapper
        make_sequential_obs = True


    else:
        sequential_wrapper_kwargs = None
        seqential_wrapper_cls = None
        make_sequential_obs = False
     

    envs = make_vec_env_robosuite(
        "PickPlaceCan",
        obs_keys = ["object-state","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        rng=np.random.default_rng(SEED),
        n_envs=12,
        parallel=True,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
        env_make_kwargs=make_env_kwargs,
        sequential_wrapper = seqential_wrapper_cls,
        sequential_wrapper_kwargs = sequential_wrapper_kwargs
    )

    annotation_dict = read_all_json("progress_data")

    trajs = load_dataset_to_trajectories(["object","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                                         dataset_path = "human-demo/can-pick/low_dim_v141.hdf5", 
                                            make_sequential_obs=make_sequential_obs,
                                         sequential_obs_keys=args.sequence_keys,
                                         obs_seq_len=args.obs_seq_len,
                                         use_half_gripper_obs=True)
    
    
    trajs_for_shaping, annotation_list = load_dataset_and_annotations_simutanously(["object","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                                                                       annotation_dict=annotation_dict,
                                                                       dataset_path=dataset_path,
                                                                       make_sequential_obs=make_sequential_obs,
                                         sequential_obs_keys=args.sequence_keys,
                                         obs_seq_len=args.obs_seq_len,
                                         use_half_gripper_obs=True)
    # type of reward shaping to use
    # change this to enable or disable reward shaping
    # shape_reward = ["progress_sign_loss", "delta_progress_scale_loss", ]
    shape_reward = []

    policy_kwargs = dict(
        manual_close_gripper_chance = 0.8)
                                                                  
    learner = PPO(
        env=envs,
        policy=ActorCriticPolicyWrapperManualCloseGripper,
        batch_size=1024,
        ent_coef=0.0,
        learning_rate=0.0005,
        gamma=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        n_epochs=10,
        seed=SEED,
        policy_kwargs=policy_kwargs
    )
    reward_net = BasicShapedRewardNet(
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        normalize_input_layer=RunningNorm,
    )
    generator_model_path = f"{project_path}/checkpoints/{args.load_exp_name}/{args.checkpoint}/gen_policy/model"
    if args.continue_training:
        reward_net = (torch.load(f"{project_path}/checkpoints/{args.load_exp_name}/{args.checkpoint}/reward_train.pt"))
        learner = PPO.load(generator_model_path)
    # logger that write tensroborad to logs dir
    logger = imit_logger.configure(folder=log_dir, format_strs=["tensorboard"])
    airl_trainer = AIRL(
        demonstrations=trajs,
        demo_batch_size=2048,
        gen_replay_buffer_capacity=1000000,
        n_disc_updates_per_round=32,
        venv=envs,
        gen_algo=learner,
        reward_net=reward_net,
        shape_reward = shape_reward,
        annotation_list=annotation_list,
        demostrations_for_shaping=trajs_for_shaping,
        custom_logger = logger,
        save_path = f"checkpoints/{args.exp_name}"
        # log_dir = log_dir,
        # init_tensorboard = True,
        # init_tensorboard_graph = True
    )

    # loss = airl_trainer.progress_shaping_loss()
    # print(loss)
    # loss.backward()
    # envs.seed(SEED)

    # load the model to continue training

    

    learner_rewards_before_training, _ = evaluate_policy(
        learner, envs, 12, return_episode_rewards=True,
    )
    airl_trainer.train(8_000_000)  # Train for 2_000_000 steps to match expert.
    # envs.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, envs, 12, return_episode_rewards=True,
    )

    print("mean reward after training:", np.mean(learner_rewards_after_training))
    print("mean reward before training:", np.mean(learner_rewards_before_training))
    # save the model
    # if not os.path.exists(os.path.join(project_path,f"checkpoints/{args.exp_name}")):
    #     os.makedirs(os.path.join(project_path,f"checkpoints/{args.exp_name}"))
    # train_adversarial.save(airl_trainer, 
    #                        os.path.join(project_path,f"checkpoints/{args.exp_name}"),
    #                        )
