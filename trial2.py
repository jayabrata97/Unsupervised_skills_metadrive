#from metadrive import MetaDriveEnv
from metadrive.envs.metadrive_env import MetaDriveEnv
#from metadrive import SafeMetaDriveEnv
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
import argparse
import random
import gym
import numpy as np
import matplotlib.pyplot as plt

traffic_density_sample = np.random.uniform(0.5, 0.9)
if traffic_density_sample < 0.5:
    traffic_density_sample = 0.5

env = MetaDriveEnv(dict(
    # controller="joystick",
    # use_render= True,
    # manual_control=True,
    traffic_density= 0.1,
    random_traffic = False, 
    environment_num=1,
    start_seed=0,
    use_lateral = True,
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

x_pos=[]
y_pos=[]

ep_reward = 0.0
obs = env.reset()
for i in range(10000000):
    obs, reward, done, info = env.step(env.action_space.sample())
    ego_position = env.vehicle.position
    x_pos.append(ego_position[0])
    y_pos.append(ego_position[1])
    print("ego position: ", ego_position) #physx_world position

    heading_theta = env.vehicle.heading_theta
    print("heading_theta: ", heading_theta)

    speed = env.vehicle.speed
    print("speed:", speed)

    velocity = env.vehicle.velocity
    print("velocity: ",velocity)

    heading_diff = env.vehicle.heading_diff
    print("heading_diff: ", heading_diff)

    current_ref_lanes = env.vehicle.navigation.current_ref_lanes
    print("current_ref_lanes: ", current_ref_lanes)

    vehicle_lane = env.vehicle.lane
    print("vehicle lane: ", vehicle_lane)

    if env.vehicle.lane in env.vehicle.navigation.current_ref_lanes:
        current_lane = env.vehicle.lane
        positive_road = 1
    else:
        current_lane = env.vehicle.navigation.current_ref_lanes[0]
        current_road = env.vehicle.navigation.current_road
        positive_road = 1 if not current_road.is_negative_road() else -1
    long_last, lateral_last = current_lane.local_coordinates(env.vehicle.last_position)  # local lane position
    long_now, lateral_now = current_lane.local_coordinates(env.vehicle.position) # local lane position
    print("long_now: ", long_now, ";lateral_now:", lateral_now)
    print("long_last: ", long_last, ";lateral_last: ", lateral_last)

    info_for_ckpt = env.vehicle.navigation._get_info_for_checkpoint(lanes_id=vehicle_lane, 
                                                                    ref_lane=env.vehicle.lane,
                                                                    ego_vehicle=env.vehicle)
    print("info_for_ckpt: ", info_for_ckpt)
    
    nav_info = env.vehicle.navigation.get_navi_info()
    print("navigation info: ", nav_info)

    # info = env.observation_space.observe(env.vehicle)
    # print("info: ", info)

    print("Current velocity: ", info["velocity"])
    print()

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
x_pos_arr = np.array(x_pos)
y_pos_arr = np.array(y_pos)
plot= plt.plot(x_pos_arr, y_pos_arr)
plt.savefig("distance.jpg")
print("\nMetaDrive successfully run!")