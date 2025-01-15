import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

# from imitation.algorithms.adversarial.airl import AIRL
from IRL_lib_mod.airl_updating import AIRL
from imitation.algorithms.adversarial.airl import AIRL as AIRL_old
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

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

print_cnt = 0

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
    parser.add_argument("--env_name", type = str, default="lift")
    parser.add_argument("--dataset_type", type=str, default="ph")
    parser.add_argument('--exp_name', type=str, default="default_experiment")
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default="320")
    parser.add_argument('--load_exp_name', type=str, default="mh_sign_scale_loss_8m_1")
    parser.add_argument('-s', '--sequence_keys', nargs='+', default=[])
    parser.add_argument('-l', '--obs_seq_len', type=int, default=1)


    
    args = parser.parse_args()
    print("sequence keys", args.sequence_keys)
    print("obs_seq_len", args.obs_seq_len)


    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/learning-with-progress"
    print(project_path)
    dataset_path = os.path.join(project_path,"human-demo/" + args.env_name +"/low_dim_v141_" 
                                + args.env_name + "_" + args.dataset_type + ".hdf5")
    #dataset_path = os.path.join(project_path,"human-demo/square/low_dim_v141.hdf5")
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
        horizon=600,                            # each episode terminates after 300 steps
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
     
    if args.env_name == "square":
        robosuite_env_name = "NutAssemblySquare"
    if args.env_name == "can-pick":
        robosuite_env_name = "PickPlaceCan"
    if args.env_name == "lift":
        robosuite_env_name = "Lift"  
    envs = make_vec_env_robosuite(
        robosuite_env_name,
        obs_keys = ["cube_pos","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        rng=np.random.default_rng(SEED),
        n_envs=5,
        parallel=True,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
        env_make_kwargs=make_env_kwargs,
        sequential_wrapper = seqential_wrapper_cls,
        sequential_wrapper_kwargs = sequential_wrapper_kwargs
    )

    annotation_dict = read_all_json(args.env_name + "_" + args.dataset_type)

    trajs = load_dataset_to_trajectories(["object","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                                         dataset_path = "human-demo/" + args.env_name +"/low_dim_v141_" 
                                            + args.env_name + "_" + args.dataset_type + ".hdf5", 
                                            make_sequential_obs=make_sequential_obs,
                                         sequential_obs_keys=args.sequence_keys,
                                         obs_seq_len=args.obs_seq_len,
                                         use_half_gripper_obs=True,
                                         use_cube_pos=True
                                         )
    
    # for i in range(len(trajs)):
    #     if trajs[i].obs.shape[1] != 31:
    #         print(trajs[i].obs.shape)

    trajs_for_shaping, annotation_list = load_dataset_and_annotations_simutanously(["object","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                                                                       annotation_dict=annotation_dict,
                                                                       dataset_path=dataset_path,
                                                                       make_sequential_obs=make_sequential_obs,
                                         sequential_obs_keys=args.sequence_keys,
                                         obs_seq_len=args.obs_seq_len,
                                         use_half_gripper_obs=True,
                                         use_cube_pos=True
                                                                       )
    # type of reward shaping to use
    # change this to enable or disable reward shaping
    #shape_reward = ["progress_sign_loss", "value_sign_loss", "advantage_sign_loss"]
    # print("**********************************************************")
    # print(envs.observation_space)
    # # print trajectory obs shape
    # print(trajs_for_shaping[0].obs.shape)
    # print("**********************************************************")
    traj_index = []
    for i in range(len(trajs_for_shaping)):
        if trajs_for_shaping[i].obs.shape[1] != 31:
            #print(trajs_for_shaping[i].obs.shape)
            traj_index.append(i)
                                                                  
    learner = PPO(
        env=envs,
        policy=CustomLoggingPolicy,  # Use your custom policy here
        #policy=MlpPolicy,
        batch_size=256,
        ent_coef=0.01,
        learning_rate=3e-4,
        gamma=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        n_epochs=10,
        seed=SEED,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        normalize_input_layer=RunningNorm,
        reward_hid_sizes=(64, 64),
        potential_hid_sizes=(64, 64),
    )
    generator_model_path = f"{project_path}/checkpoints/{args.load_exp_name}/{args.checkpoint}/gen_policy/model"
    if args.continue_training:
        reward_net = (torch.load(f"{project_path}/checkpoints/{args.load_exp_name}/{args.checkpoint}/reward_train.pt"))
        learner = PPO.load(generator_model_path)
    # logger that write tensroborad to logs dir
    logger = imit_logger.configure(folder=log_dir, format_strs=["tensorboard"])
# all available shaping types are ["value_sign_loss", "advantage_sign_loss", "progress_sign_loss",
# "delta_progress_scale_loss", "progress_value_loss", "value_sign_loss_alternative",
# "progress_sign_loss_alternative", "demo_range_loss", "progress_regression_loss", "progress_head_loss",
# "progress_regularization"]
    shape_reward = [
        #"demo_range_loss",
        #"delta_progress_scale_loss",
        "advantage_sign_loss",
        "value_sign_loss",
        "reward_sign_loss",
        #"subtrajectory_proportion_loss"
    ]
    #print("trajectory for shaping:", len(trajs_for_shaping))
    airl_trainer = AIRL(
        demonstrations=trajs,
        demo_batch_size=32,
        gen_replay_buffer_capacity=20000,
        n_disc_updates_per_round=10,
        venv=envs,
        gen_algo=learner,
        reward_net=reward_net,
        shape_reward = shape_reward,
        annotation_list=annotation_list,
        demostrations_for_shaping=trajs_for_shaping,
        custom_logger = logger,
        save_path = f"checkpoints/{args.exp_name}",
        shaping_batch_size=16,
        traj_index = traj_index,
        allow_variable_horizon=True
    )
# trainer = AIRLWithProgress(
#     demonstrations=my_demos,
#     demo_batch_size=32,
#     venv=my_venv,
#     gen_algo=my_rl_algorithm,
#     reward_net=reward_net,
#     annotation_list=my_annotation_list,
#     demonstrations_for_shaping=my_trajs_for_shaping,
#     shape_reward=[
#         "demo_range_loss",
#         "progress_regression_loss",
#         "progress_head_loss",
#         "progress_regularization"
#     ],
#     shaping_batch_size=16,
#     shaping_loss_weight=1.0,
#     alpha=1.0,
#     lambda_reg=0.1,
#     progress_head_weight=0.05
# )
    start = 0
    training_time = 1_000_000
    training_round = 50
    record_file = f"{project_path}/checkpoints/log_files/" + args.exp_name + ".txt"
    import time
    start_time = time.time()
    for i in range(start, training_round):
        airl_trainer.train(total_timesteps=training_time)
        if i % 1 == 0:
            elapsed_time_seconds = time.time() - start_time
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))
            learner_rewards = evaluate_policy(learner, envs, n_eval_episodes=10)
            print("learner mean reward at round", i, ":", np.mean(learner_rewards))
            print("running time ", i, "round :", elapsed_time_str)
            with open(record_file, "a") as f:
                f.write("mean reward at round " + str(i) + ":" + str(np.mean(learner_rewards)) + "\n")
                f.write("running time " + str(i) + " round :" + elapsed_time_str + "\n")
            f.close()