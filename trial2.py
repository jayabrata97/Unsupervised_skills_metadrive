from metadrive import MetaDriveEnv
from metadrive import SafeMetaDriveEnv
import argparse
import random
import gym
import numpy as np

traffic_density_sample = np.random.uniform(0.5, 0.9)
if traffic_density_sample < 0.5:
    traffic_density_sample = 0.5

env = MetaDriveEnv(dict(
    # controller="joystick",
    use_render= True,
    manual_control=True,
    traffic_density= 0.1,
    random_traffic = False, 
    environment_num=200,
    start_seed=0,
    #start_seed=random.randint(0, 1000)
    random_lane_width=False,
    random_agent_model=True,
    random_lane_num=False,
    map=7
))

# env = gym.make("MetaDrive-validation-v0")
# env = MetaDriveEnv(dict(environment_num=1000))  # Or you can also choose to create env from class.

print("\nThe action space: {}".format(env.action_space))
print("\nThe observation space: {}\n".format(env.observation_space))
print("Starting the environment ...\n")

ep_reward = 0.0
obs = env.reset()
for i in range(100000000):
    obs, reward, done, info = env.step(env.action_space.sample())
    #print("obs type:",type(obs))
    #print("step reward:",reward)
    #print("Current velocity: ", info["velocity"])
    ep_reward += reward
    if done:
        break
    #if done:
    # if (i%100) == 0:
    #     print("\nThe episode reward: ", ep_reward)
    #     print("\nThe returned information: {}.".format(info))
    #     obs = env.reset()
        #break

print("\nThe observation shape: {}.".format(obs.shape))
print("\nThe returned episode reward: {}.".format(ep_reward))
print("\nThe returned information: {}.".format(info))
print('i:', i)

env.close()
print("\nMetaDrive successfully run!")