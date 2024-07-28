# replay trajectory from dataset and collect progress data from user
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
import cv2

ENV_META_EXCLUDE = ["env_version", "type"]

def replay_trajectory_and_collect_progress(dataset_path:str,
                      reply_demo_indicies:int,
                      collect_progress_times:int, 
                      **env_kwargs):
    '''
    replay trajectory from dataset and collect progress data from user
    write progress data to json file
    input: dataset_path: str, relative to human_demo folder
              reply_demo_numbers: int, number of demo to replay
              collect_progress_times: int, number of times to collect progress data
     '''
    # load dataset
    f:h5py.File = load_data_to_h5py(dataset_path)

    # get environment meta data
    env_name = json.loads(f["data"].attrs["env_args"])["env_name"]
    env_kwargs = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
    # enable rendering
    env_kwargs["has_renderer"] = True

    # make environment
    env:PickPlaceCan = suite.make(
        env_name=env_name,
        **env_kwargs
    )

    # get demo keys
    filter_key = "train"
    demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)][:])]
    # print(demo_keys)
    # get demo keys to replay
    print("replay_demo:",reply_demo_indicies)  
    replay_demo_keys = ["demo_{}".format(i) for i in reply_demo_indicies if "demo_{}".format(i) in demo_keys]
    # print(replay_demo_keys)
    # replay demo, pause given times and collect progress data
    progress_data = dict()
    for key in replay_demo_keys:
        print(key)
        obs = np.array(f["data/{}/obs".format(key)])
        actions = np.array(f["data/{}/actions".format(key)])
        print(actions)
        dones = np.array(f["data/{}/dones".format(key)])
        # set initial state
        initial_state = f["data/{}/states".format(key)][0]

        env.reset()
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()
        env.render()
        

        pause_indices = np.linspace(0, len(actions), collect_progress_times+2, dtype=int)[1:-1]
        pause_indices = np.append(pause_indices, len(actions)-1)
        progress = [0]
        # replay demo
        for i in range(len(actions)):
            action = actions[i]
            # obs = obs[i]
            done = dones[i]
            env.step(action)
            env.render()
            if i == 0:
                time.sleep(1)
            # if done:
            #     break
            
            # pause and collect progress data
            if i in pause_indices:
                progress_data[key] = progress_data.get(key, [])
                # get user input
                user_input = input("Please input the progress data: ")
                # user input must be a float, otherwise ask user to input again
                while not user_input.replace(".", "").isdigit():
                    user_input = input("Please input the progress data: ")
                progress.append(float(user_input))
                
                single_data = dict(
                    start_step = int(pause_indices[np.where(pause_indices == i)[0][0]-1] if np.where(pause_indices == i)[0][0]-1 >= 0 else 0),
                    end_step = i,
                    start_progress = progress[len(progress)-2] if len(progress) >= 2 else 0,
                    end_progress = float(user_input)
                )

                progress_data[key].append(single_data)
                # render the environment
                env.render()
            if i == len(actions)-1:
                break
    # write progress data to json file, each demo has a json file
        for key in progress_data.keys():
            write_to_json(progress_data[key], "{}.json".format(key))
    
    f.close()


def replay_trajectory_and_collect_preference(dataset_path:str,
                      replay_demo_nums:int,
                      collect_preference_times:int, 
                      demo_quality_choise = "random",
                      **env_kwargs):
    '''
    replay trajectory from dataset and collect progress data from user
    write progress data to json file
    input: dataset_path: str, relative to human_demo folder
              reply_demo_numbers: int, number of demo to replay
              collect_progress_times: int, number of times to collect progress data
     '''
    # load dataset
    f:h5py.File = load_data_to_h5py(dataset_path)

    # get environment meta data
    env_name = json.loads(f["data"].attrs["env_args"])["env_name"]
    env_kwargs = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
    # enable rendering
    env_kwargs["has_renderer"] = True
    
    # get camera obs and play them as a bypass for "multiple rendering"
    env_kwargs["camera_heights"] = 2048
    env_kwargs["camera_widths"] = 2048
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True

    # make environments, we make two envs since the preference collection need to watch 
    # two demos at same time
    env_1:PickPlaceCan = suite.make(
        env_name=env_name,
        **env_kwargs
    )
    env_2:PickPlaceCan = suite.make(
        env_name=env_name,
        **env_kwargs
    )

    # get demo names
    better_demo_key = "better"
    okay_demo_key = "okay"
    worse_demo_key = "worse"

    better_demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(better_demo_key)][:])]
    okay_demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(okay_demo_key)][:])]
    worse_demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(worse_demo_key)][:])]

    # get pairs of demo keys to replay
    replay_demo_key_pairs = generate_demo_pair_for_preference(
        demo_choose_method=demo_quality_choise, 
        replay_demo_nums = replay_demo_nums,
        better_demos=better_demos, 
        okay_demos=okay_demos,
        worse_demos=worse_demos
    )
    # replay demos according to key pairs, collect preference
    

    replay_demo_key_pairs = [("demo_1","demo_200"), ("demo_2", "demo_199")]

    # create windows for replaying, we use opencv to create two windows at same time
    cv2.namedWindow('Window 1')
    cv2.namedWindow('Window 2')

    for (key_1, key_2) in replay_demo_key_pairs:
        preferences = []

        print(f"replaying: {key_1} and {key_2}")
        obs_1 = np.array(f["data/{}/obs".format(key_1)])
        actions_1 = np.array(f["data/{}/actions".format(key_1)])
        dones_1 = np.array(f["data/{}/dones".format(key_1)])
        # set initial state
        initial_state_1 = f["data/{}/states".format(key_1)][0]

        obs_2 = np.array(f["data/{}/obs".format(key_2)])
        actions_2 = np.array(f["data/{}/actions".format(key_2)])
        dones_2 = np.array(f["data/{}/dones".format(key_2)])
        initial_state_2 = f["data/{}/states".format(key_2)][0]

        # set initial state for both envs
        env_1.reset()
        env_1.sim.set_state_from_flattened(initial_state_1)
        env_1.sim.forward()
        # env_1.render()

        env_2.reset()
        env_2.sim.set_state_from_flattened(initial_state_2)
        env_2.sim.forward()
        # env_2.render()
        
        # get_pause_indicies
        pause_indices_1 = np.linspace(0, len(actions_1), collect_preference_times+1, dtype=int)
        pause_indices_2 = np.linspace(0, len(actions_2), collect_preference_times+1, dtype=int)

        # replay demo
        for sub_traj_idx in range(collect_preference_times):
            
            imgs_1 = []
            imgs_2 = []

            # replay demo_1 subtraj
            for i_1 in range(pause_indices_1[sub_traj_idx], pause_indices_1[sub_traj_idx+1]):
                action = actions_1[i_1]
                obs,_,_,_ = env_1.step(action)
                # filp the image
                imgs_1.append(cv2.flip(obs["agentview_image"], 0))
                # env_1.render()
                # pause on first frame

            # replay demo_2 subtraj
            for i_2 in range(pause_indices_2[sub_traj_idx], pause_indices_2[sub_traj_idx+1]):
                action = actions_2[i_2]
                obs,_,_,_ = env_2.step(action)
                imgs_2.append(cv2.flip(obs["agentview_image"], 0))
                # env_2.render()
                # pause on first frame
            
            # play the "recorded video" for subtraj
            display_images(imgs_1, "Window 1")
            display_images(imgs_2, "Window 2")


            user_preference = int(input("Please input the preference data, 1 or 2:"))
            preference_data = dict(
                start_frame_1 = int(pause_indices_1[sub_traj_idx]),
                end_frame_1 = int(pause_indices_1[sub_traj_idx+1]),
                start_frame_2 = int(pause_indices_2[sub_traj_idx]),
                end_frame_2 = int(pause_indices_2[sub_traj_idx+1]),
                demo_1 = key_1,
                demo_2 = key_2,
                preference = user_preference
            )
            preferences.append(preference_data)

        # write preference data to json file
        write_to_json(preferences, "preference_{}_{}.json".format(key_1, key_2), data_folder="preference_data")
        print("replay finished for pair: {} and {}".format(key_1, key_2))
    
    # write progress data to json file, each demo has a json file
    #     for key in preference_data.keys():
    #         write_to_json(preference_data[key], "preference_{}.json".format(key))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    f.close()


def generate_demo_pair_for_preference(demo_choose_method,
                                      replay_demo_nums,
                                      better_demos,
                                      okay_demos, 
                                      worse_demos):
    pass

def display_images(img_list, window_name, fps=20):
    frame_time = int(1000 / fps)  # Calculate time to wait for each image in milliseconds
    
    for id, img in enumerate(img_list):
        # ask user to press any key to continue if play the first frame
        if id == 0:
            cv2.imshow(window_name, cv2.resize(img, (256,256)))
            input("Press any key to start playing the video")
        if img is not None:
            resized_img = cv2.resize(img, (256,256))
            cv2.imshow(window_name, resized_img)
            cv2.waitKey(frame_time)  # Display each image for the calculated time
        else:
            print(f"Failed to load image: {img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_path", type=str, default="can-pick/low_dim_v141.hdf5")
    parser.add_argument("--dataset_path", type=str, default="can-pick/can_low_dim_mh.hdf5")

    parser.add_argument("--replay_demo_numbers", type=int, nargs="+", default=[1])
    parser.add_argument("--collect_progress_times", type=int, default=10)
    args = parser.parse_args()
    print(args.replay_demo_numbers)
    replay_trajectory_and_collect_preference(args.dataset_path, args.replay_demo_numbers, args.collect_progress_times)
