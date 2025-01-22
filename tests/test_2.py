import robosuite as suite

env = suite.make(
    robots="Panda",  # use panda robot
    env_name="PickPlaceCan",  # or whichever environment
)

obs = env.reset()
print(obs.keys())
