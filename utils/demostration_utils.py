# adapt dataset to library
from typing import Iterable
from imitation.data import types
import numpy as np
import h5py
import os

def load_dataset_to_trajectories(obs_keys:Iterable[str],
                                 dataset_path:str = "human-demo/can-pick/low_dim_v141.hdf5",):
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,dataset_path)
    f = h5py.File(dataset_path,'r')
    filter_key = "train"
    demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)][:])]
    trajectories = []
    for key in demo_keys:
        obs_dict = {obs_key: np.array(f["data/{}/obs/{}".format(key, obs_key)]) for obs_key in obs_keys}
        # print(obs_dict.keys())
        # print(obs_dict["robot0_eef_pos"])
        zipped = zip(*obs_dict.values()) 
        obs = [np.concatenate(elem, axis=0) for elem in zipped]
        obs = np.stack(obs, axis=0)
        # print(obs)
        # print(len(obs))
        # print(obs[0].shape)
        # actions need to be cut off the last element
        action = np.array(f["data/{}/actions".format(key)][:-1])
        dones = np.array(f["data/{}/dones".format(key)][-1])
        trajectory = types.Trajectory(obs=obs, acts=action, terminal=dones, infos=None)
        trajectories.append(trajectory)
    return trajectories

def load_dataset_and_annotations_simutanously(obs_keys:Iterable[str],
                                                annotation_dict:dict,
                                                dataset_path:str = "human-demo/can-pick/low_dim_v141.hdf5"):

    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,dataset_path)
    f = h5py.File(dataset_path,'r')
    filter_key = "train"
    demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)][:])]
    trajectories = []
    annotation_list = []
    for key in demo_keys:

        # load obs
        obs_dict = {obs_key: np.array(f["data/{}/obs/{}".format(key, obs_key)]) for obs_key in obs_keys}
        zipped = zip(*obs_dict.values()) 
        obs = [np.concatenate(elem, axis=0) for elem in zipped]
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
        annotation_list.append(annotation)
    return trajectories, annotation_list



def load_data_to_h5py(dataset_path: str):
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_path,"human-demo",dataset_path)
    f = h5py.File(dataset_path,'r')
    return f

