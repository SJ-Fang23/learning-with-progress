import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

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
            total_actions += actions_np.shape[0]
            positive_last = np.sum(actions_np[:, -1] > 0)
            ratio = positive_last / actions_np.shape[0]
            print(f"Positive ratio: {ratio}")

        # Log the actions (you can adjust the logging as needed)
        
        
        # Return the outputs as usual
        return actions, values, log_probs

# Define the custom callback
class ResetPPOCallback(BaseCallback):
    """
    Callback for resetting the PPO learner every reset_interval timesteps.
    """
    def __init__(self, reset_interval: int, airl_trainer, verbose=0):
        super(ResetPPOCallback, self).__init__(verbose)
        self.reset_interval = reset_interval
        self.airl_trainer = airl_trainer
        self.total_timesteps = 0

    def _on_step(self) -> bool:
        self.total_timesteps += self.n_calls  # n_calls is usually 1 per step
        if self.total_timesteps >= self.reset_interval:
            if self.verbose > 0:
                print(f"Resetting PPO learner at timestep {self.total_timesteps}")
            # Reinitialize the PPO learner
            self.airl_trainer.gen_algo = PPO(
                policy=MlpPolicy,
                env=self.airl_trainer.venv,
                batch_size=32,
                ent_coef=0.01,
                learning_rate=3e-4,
                gamma=0.95,
                clip_range=0.2,
                vf_coef=0.5,
                n_epochs=10,
                seed=42,
            )
            self.total_timesteps = 0  # Reset the counter
        return True


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
    dataset_path = os.path.join(project_path,"human-demo/can-pick/can_low_dim_mh.hdf5")
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
        horizon=500,                            # each episode terminates after 300 steps
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

    annotation_dict = read_all_json("progress_data_mh")

    trajs = load_dataset_to_trajectories(["object","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                                         dataset_path = "human-demo/can-pick/can_low_dim_mh.hdf5", 
                                            make_sequential_obs=make_sequential_obs,
                                         sequential_obs_keys=args.sequence_keys,
                                         obs_seq_len=args.obs_seq_len,
                                         use_half_gripper_obs=True
                                         )
    
    for i in range(len(trajs)):
        if trajs[i].obs.shape[1] != 31:
            print(trajs[i].obs.shape)

    trajs_for_shaping, annotation_list = load_dataset_and_annotations_simutanously(["object","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                                                                       annotation_dict=annotation_dict,
                                                                       dataset_path=dataset_path,
                                                                       make_sequential_obs=make_sequential_obs,
                                         sequential_obs_keys=args.sequence_keys,
                                         obs_seq_len=args.obs_seq_len,
                                         use_half_gripper_obs=True)
    # type of reward shaping to use
    # change this to enable or disable reward shaping
    #shape_reward = ["progress_sign_loss", "value_sign_loss", "advantage_sign_loss"]
    shape_reward = []

    for i in range(len(trajs_for_shaping)):
        if trajs_for_shaping[i].obs.shape[1] != 31:
            print(i)
                                                                  
    learner = PPO(
        env=envs,
        #policy=CustomLoggingPolicy,  # Use your custom policy here
        policy=MlpPolicy,
        batch_size=32,
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
    airl_trainer = AIRL(
        demonstrations=trajs,
        demo_batch_size=64,
        gen_replay_buffer_capacity=10000,
        n_disc_updates_per_round=2,
        venv=envs,
        gen_algo=learner,
        reward_net=reward_net,
        shape_reward = shape_reward,
        annotation_list=annotation_list,
        demostrations_for_shaping=trajs_for_shaping,
        custom_logger = logger,
        save_path = f"checkpoints/{args.exp_name}",
    )

    # loss = airl_trainer.progress_shaping_loss()
    # print(loss)
    # loss.backward()
    # envs.seed(SEED)

    # load the model to continue training


################################################################
    # check the gripper ever closed

################################################################    

    # learner_rewards_before_training, _ = evaluate_policy(
    #     learner, envs, 12, return_episode_rewards=True,
    # )

    # Train the PPO with the function-based callback that logs actions
    # Initialize the custom callback
    reset_callback = ResetPPOCallback(
        reset_interval=1000,  # Reset every 120,000 timesteps
        airl_trainer=airl_trainer,
        verbose=1
    )

    # Train the AIRL trainer with the callback
    airl_trainer.train(
        total_timesteps=12_000_000,
        callback=reset_callback
    )
# envs.seed(SEED)
    # learner_rewards_after_training, _ = evaluate_policy(
    #     learner, envs, 12, return_episode_rewards=True,
    # )

    # print("mean reward after training:", np.mean(learner_rewards_after_training))
    # print("mean reward before training:", np.mean(learner_rewards_before_training))
    # # save the model
    # if not os.path.exists(os.path.join(project_path,f"checkpoints/{args.exp_name}")):
    #     os.makedirs(os.path.join(project_path,f"checkpoints/{args.exp_name}"))
    # train_adversarial.save(airl_trainer, 
    #                        os.path.join(project_path,f"checkpoints/{args.exp_name}"),
    #                        )
