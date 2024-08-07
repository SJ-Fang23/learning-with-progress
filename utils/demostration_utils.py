# adapt dataset to library
from typing import Iterable
from imitation.data import types
import numpy as np
import h5py
import os

def load_dataset_to_trajectories(obs_keys:Iterable[str],
                                 dataset_path:str = "human-demo/can-pick/low_dim_v141.hdf5",
                                 make_sequential_obs:bool = False, 
                                 sequential_obs_keys:Iterable[str] = None, 
                                 obs_seq_len:int = 1, 
                                 use_half_gripper_obs = True):
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,dataset_path)
    f = h5py.File(dataset_path,'r')
    filter_key = "train"
    demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)][:])]
    trajectories = []
    for key in demo_keys:
        
        # get the normal not sequential observations part
        obs = [np.array(f["data/{}/obs/{}".format(key, obs_key)]) for obs_key in obs_keys if not obs_key in sequential_obs_keys]

        # get the sequential observations part
        if make_sequential_obs:
            if use_half_gripper_obs:
                seq_obs = []
                # seq_obs = [np.array(f["data/{}/obs/{}".format(key, obs_key)][:, 0:1]) for obs_key in sequential_obs_keys]
                for obs_key in sequential_obs_keys:
                    if obs_key == "robot0_gripper_qpos":
                        seq_obs.append(np.array(f["data/{}/obs/{}".format(key, obs_key)][:, 0:1]))
                    else:
                        seq_obs.append(np.array(f["data/{}/obs/{}".format(key, obs_key)]))
            else:
                seq_obs = [np.array(f["data/{}/obs/{}".format(key, obs_key)]) for obs_key in sequential_obs_keys]
            seq_obs = np.concatenate(seq_obs, axis=1)
            seq_obs = np.stack(seq_obs, axis=0)
            # pad the first obs_seq_len - 1 elements with first element
            seq_obs = np.concatenate([np.repeat(seq_obs[0:1], obs_seq_len - 1, axis=0), seq_obs], axis=0)

            # make the sequential obs
            seq_obs = [np.concatenate(seq_obs[i:i+obs_seq_len], axis=0) for i in range(len(seq_obs) - obs_seq_len + 1)]
            # add the sequential obs to the normal obs
            obs.append(seq_obs)

        # concatenate all the observations
        obs = np.concatenate(obs, axis=1)

        # stack the observations
        obs = np.stack(obs, axis=0)
        # print(obs[-1].shape)
        # actions need to be cut off the last element
        action = np.array(f["data/{}/actions".format(key)][:-1])
        dones = np.array(f["data/{}/dones".format(key)][-1])
        trajectory = types.Trajectory(obs=obs, acts=action, terminal=dones, infos=None)
        trajectories.append(trajectory)
    return trajectories

def load_dataset_and_annotations_simutanously(obs_keys:Iterable[str],
                                                annotation_dict:dict,
                                                dataset_path:str = "human-demo/can-pick/low_dim_v141.hdf5", 
                                                make_sequential_obs:bool = False, 
                                                sequential_obs_keys:Iterable[str] = None, 
                                                obs_seq_len:int = 1, 
                                                use_half_gripper_obs = True):

    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,dataset_path)
    f = h5py.File(dataset_path,'r')
    filter_key = "train"
    demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)][:])]
    annotation_keys = list(annotation_dict.keys())
    trajectories = []
    annotation_list = []
    for key in annotation_keys:

        # load obs
        # get the normal not sequential observations part
        obs = [np.array(f["data/{}/obs/{}".format(key, obs_key)]) for obs_key in obs_keys if not obs_key in sequential_obs_keys]

        # get the sequential observations part
        if make_sequential_obs:
            if use_half_gripper_obs:
                seq_obs = []
                # seq_obs = [np.array(f["data/{}/obs/{}".format(key, obs_key)][:, 0:1]) for obs_key in sequential_obs_keys]
                for obs_key in sequential_obs_keys:
                    if obs_key == "robot0_gripper_qpos":
                        seq_obs.append(np.array(f["data/{}/obs/{}".format(key, obs_key)][:, 0:1]))
                    else:
                        seq_obs.append(np.array(f["data/{}/obs/{}".format(key, obs_key)]))
            else:
                seq_obs = [np.array(f["data/{}/obs/{}".format(key, obs_key)]) for obs_key in sequential_obs_keys]
            seq_obs = np.concatenate(seq_obs, axis=1)
            seq_obs = np.stack(seq_obs, axis=0)
            # pad the first obs_seq_len - 1 elements with first element
            seq_obs = np.concatenate([np.repeat(seq_obs[0:1], obs_seq_len - 1, axis=0), seq_obs], axis=0)

            # make the sequential obs
            seq_obs = [np.concatenate(seq_obs[i:i+obs_seq_len], axis=0) for i in range(len(seq_obs) - obs_seq_len + 1)]
            # add the sequential obs to the normal obs
            obs.append(seq_obs)

        # concatenate all the observations
        obs = np.concatenate(obs, axis=1)

        obs = np.stack(obs, axis=0)

        # load actions
        action = np.array(f["data/{}/actions".format(key)][:-1])
        dones = np.array(f["data/{}/dones".format(key)][-1])

        # create trajectory
        trajectory = types.Trajectory(obs=obs, acts=action, terminal=dones, infos=None)
        trajectories.append(trajectory)

        # get corresponding annotation
        annotation = annotation_dict[key]
        # zip every elemet with corresponding demostration index (number of corresponding trajectory)
        annotation = [(elem, len(trajectories) - 1) for elem in annotation]
        annotation_list.extend(annotation)
    return trajectories, annotation_list



def load_data_to_h5py(dataset_path: str):
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,"human-demo",dataset_path)
    f = h5py.File(dataset_path,'r')
    return f

