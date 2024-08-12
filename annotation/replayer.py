# replay trajectories from dataset and observe
import h5py
import os
import json
import robosuite as suite
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.pickplace import*
import numpy as np
from robosuite.controllers import load_controller_config
import argparse
from utils.demostration_utils import load_data_to_h5py
from utils.annotation_utils import write_to_json
from robosuite.wrappers import GymWrapper



def replay_demo(dataset_path: str):

    # project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    f = load_data_to_h5py(dataset_path)
    env_kwargs = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
    env_kwargs["has_renderer"] = True

    env:PickPlaceCan = suite.make(
        env_name="PickPlaceCan",
        **env_kwargs,
    )
    env = GymWrapper(env,["object-state","robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"])

    filter_key = "train"
    demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)][:])]
    # print(demo_keys)
    # get demo keys to replay

    reply_demo_indicies = [i for i in range(len(demo_keys))]

    replay_demo_keys = ["demo_{}".format(i) for i in reply_demo_indicies if "demo_{}".format(i) in demo_keys]
    
    for key in replay_demo_keys:
        actions = np.array(f["data/{}/actions".format(key)])
        dones = np.array(f["data/{}/dones".format(key)])
        initial_state = f["data/{}/states".format(key)][0]

        env.reset()
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()
        env.render()

        for i in range(len(actions)):
            action = actions[i]
            obs,_,_,_, _ =  env.step(action)
            # print(obs["object-state"][7:10])
            # print(np.linalg.norm(obs["object-state"][7:10]))
            # print(obs["robot0_eef_pos"][2])
            # print(obs[16])
            print(np.linalg.norm(obs[7:10]))
            # print("gripper action: ", action[-1])
            env.render()
            time.sleep(0.1)
            if dones[i]:
                break


if __name__ == "__main__":
    dataset_path = "can-pick/low_dim_v141.hdf5"
    replay_demo(dataset_path)
