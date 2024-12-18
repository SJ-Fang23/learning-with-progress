import h5py
import os
import json
import robosuite as suite
from envs.pickplace import*
import numpy as np
from robosuite.controllers import load_controller_config
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_path,"human-demo/square/low_dim_v141.hdf5")
print(dataset_path)
f = h5py.File(dataset_path,'r')

config_path = os.path.join(project_path,"configs/osc_position.json")
with open(config_path, 'r') as cfg_file:
    configs = json.load(cfg_file)

controller_config = load_controller_config(default_controller="OSC_POSE")
env_meta = json.loads(f["data"].attrs["env_args"])
print("env_meta: ",env_meta)
# set env to robosuite square
env =  suite.make(
    env_name="NutAssemblySquare",
    robots="Panda",             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=env_meta["env_kwargs"]["controller_configs"],   # each arm is controlled using OSC
    has_renderer=True,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                   # no observations needed
    use_camera_obs=False,
    reward_shaping=True,

)
# env:PickPlaceCan = suite.make(
#     env_name="PickPlaceCan",
#     robots="Panda",             # load a Sawyer robot and a Panda robot
#     gripper_types="default",                # use default grippers per robot arm
#     controller_configs=env_meta["env_kwargs"]["controller_configs"],   # each arm is controlled using OSC
#     has_renderer=True,                      # on-screen rendering
#     render_camera="frontview",              # visualize the "frontview" camera
#     has_offscreen_renderer=False,           # no off-screen rendering
#     control_freq=20,                        # 20 hz control for applied actions
#     horizon=200,                            # each episode terminates after 200 steps
#     use_object_obs=True,                   # no observations needed
#     use_camera_obs=False, 
# )
# print(**env_meta["env_kwargs"])

print(configs)
# print(f.keys())
# print(f["mask"].keys())
# print(f["data"].keys())
print(f["data"]["demo_0"].keys())
# print(f["data"]["demo_0"]["states"].keys())
print(f["data"]["demo_0"]["obs"].keys())
print(len(f["data"]["demo_0"]["actions"]))
print(f["data"]["demo_0"]["actions"][-1])
print(f["data"]["demo_0"]["actions"][-2])

print("obs: ",f["data"]["demo_0"]["obs"]["object"][0])
env_meta = json.loads(f["data"].attrs["env_args"])
# print(env_meta)
# print(f["data"]["demo_0"]["obs"]["robot0_eef_pos"][0])
filter_key = "train"
demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)][:])]
# print(demo_keys)
# print(len(list(f["data"]["demo_0"]["obs"])))
# print(f["data"]["demo_0"]["obs"]["object"])

initial_position = np.array(f["data"]["demo_0"]["obs"]["robot0_eef_pos"][0])


# initial_state = f["data/demo_0/states"][0]
obs = env.reset()
# # xml = env.edit_model_xml(initial_state["model"])
# # env.reset_from_xml_string(xml)
# env.sim.set_state_from_flattened(initial_state)
# env.sim.forward()
# env.render()
# # env.render()
# # for i in range(100):
# #     action = np.concatenate([initial_position,np.array([0])])
# #     obs,_,_,_ = env.step(action)
# #     env.render()

# # state_array = np.array(f["data"]["demo_0"]["obs"]["robot0_eef_pos"])
for i in range(len(f["data"]["demo_0"]["actions"])):
    action = np.array(f["data"]["demo_0"]["actions"][i])
    # action = np.concatenate([state_array[i],np.array([0])])
    obs,_,_,_ = env.step(action)
    env.render()