from envs.pickplace import PickPlace, PickPlaceCan
import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np

# import os
# os.environ["MUJOCO_GL"] = "osmesa"

controller_config = load_controller_config(default_controller="OSC_POSE")
env:PickPlace = suite.make(
    env_name="PickPlaceCan",
    robots="Panda",             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    has_renderer=True,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                   # no observations needed
    use_camera_obs=False, 
)


obs = env.reset()
print(obs)
# env.render()

# for i in range(100):
#     action = np.zeros(7)
#     obs,_,_,_ = env.step(action)
#     env.render()