from metadrive import MetaDriveEnv
from metadrive import SafeMetaDriveEnv
import argparse
import random
#import gym
import numpy as np

traffic_density_sample = np.random.uniform(0.5, 0.9)
if traffic_density_sample < 0.5:
    traffic_density_sample = 0.5

env = MetaDriveEnv(dict(
    # controller="joystick",
    use_render=True,
    # manual_control=True,
    traffic_density= np.random.uniform(0.5, 0.9), #0.1
    random_traffic = False, 
    environment_num=1000,
    start_seed=1000,
    #start_seed=random.randint(0, 1000)
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True,
    map=7
))

# env = gym.make("MetaDrive-100envs-v0")
# env = MetaDriveEnv(dict(environment_num=1000))  # Or you can also choose to create env from class.

print("\nThe action space: {}".format(env.action_space))
print("\nThe observation space: {}\n".format(env.observation_space))
print("Starting the environment ...\n")

ep_reward = 0.0
obs = env.reset()
for i in range(10000):
    obs, reward, done, info = env.step(env.action_space.sample())
    #print("obs type:",type(obs))
    print("step reward:",reward, end='\r')
    ep_reward += reward
    #if done:
    if (i%100) == 0:
        print("\nThe episode reward: ", ep_reward)
        print("\nThe returned information: {}.".format(info))
        obs = env.reset()
        #break

print("\nThe observation shape: {}.".format(obs.shape))
print("\nThe returned reward: {}.".format(reward))
print("\nThe returned information: {}.".format(info))

env.close()
print("\nMetaDrive successfully run!")