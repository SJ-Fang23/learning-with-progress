import h5py
import os
from utils.demostration_utils import load_dataset_to_trajectories
from utils.irl_utils import make_vec_env_robosuite
from imitation.data.wrappers import RolloutInfoWrapper
from envs.wrappers import SequentialObservationWrapper
import numpy as np
import json
from utils.demostration_utils import load_dataset_and_annotations_simutanously
from utils.annotation_utils import read_all_json

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,"human-demo/can-pick/can_low_dim_mh.hdf5")
    f = h5py.File(dataset_path,'r')
    trajs = load_dataset_to_trajectories(["object","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                                         dataset_path = "human-demo/can-pick/can_low_dim_mh.hdf5", 
                                         make_sequential_obs=True, 
                                         sequential_obs_keys=["robot0_gripper_qpos"], 
                                         obs_seq_len=10, 
                                         use_half_gripper_obs=True)
    
    for i in range(len(trajs)):
        if trajs[i].obs.shape[1] != 31:
            print(i)
    
    annotation_dict = read_all_json("progress_data_mh")
    trajs_for_shaping, annotation_list = load_dataset_and_annotations_simutanously(["object","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                                                                       annotation_dict=annotation_dict,
                                                                       dataset_path=dataset_path,
                                                                       make_sequential_obs=True, 
                                         sequential_obs_keys=["robot0_gripper_qpos"], 
                                         obs_seq_len=5)
    # print(annotation_list)
    
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

    sequential_wrapper_kwargs = dict(
        sequential_observation_keys = ["robot0_gripper_qpos"], 
        sequential_observation_length = 5, 
        use_half_gripper_obs = True
    )

    SEED = 42

    envs = make_vec_env_robosuite(
        "PickPlaceCanModified",
        obs_keys = ["object-state","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        rng=np.random.default_rng(SEED),
        n_envs=12,
        parallel=True,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
        env_make_kwargs=make_env_kwargs,
        sequential_wrapper = SequentialObservationWrapper,
        sequential_wrapper_kwargs = sequential_wrapper_kwargs
    )
    obs = envs.reset()
    # print(obs)

    # for _ in range(100):
    #     action = np.zeros((12, 7))
    #     # set the 7th dimension to 0.04
    #     action[:, 6] = 1
    #     obs, reward, done, info = envs.step(action)
    #     print(obs[0].shape)


    