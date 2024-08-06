import h5py
import os
from utils.demostration_utils import load_dataset_to_trajectories
from utils.irl_utils import make_vec_env_robosuite
from imitation.data.wrappers import RolloutInfoWrapper
import numpy as np
import json

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,"human-demo/can-pick/can_low_dim_mh.hdf5")
    f = h5py.File(dataset_path,'r')
    trajs = load_dataset_to_trajectories(["object","robot0_eef_pos", "robot0_eef_quat"],
                                         dataset_path = "human-demo/can-pick/can_low_dim_mh.hdf5", 
                                         make_sequential_obs=True, 
                                         sequential_obs_keys=["robot0_gripper_qpos"], 
                                         obs_seq_len=5)
    
    env_meta = json.loads(f["data"].attrs["env_args"])
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


    envs = make_vec_env_robosuite(
        "PickPlaceCanModified",
        obs_keys = ["object-state","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        # rng=np.random.default_rng(SEED),
        n_envs=12,
        parallel=True,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
        env_make_kwargs=make_env_kwargs,

    )
    