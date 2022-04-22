## Evaluation code for DADS on metadrive

import gym
import numpy as np
from scipy import stats
# from gym.wrappers import RescaleAction
from metadrive_buffer_v3 import DadsBuffer
from metadrive_networks_v3 import *
from metadrive_ppo_agent_v3 import PPOAgent
from metadrive import MetaDriveEnv
#from metadrive import SafeMetaDriveEnv
from metadrive.constants import TerminationState
import torch as T
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time

device_ids = [2,3]
device_1 = f'cuda:{device_ids[0]}'
device_2 = f'cuda:{device_ids[1]}'

def sample_skills(skill_dim):
    a = np.array([-1.0, 0.0, 1.0])
    pk = ((1/3), (1/3), (1/3))
    custm = stats.rv_discrete(name='custm', values=(a, pk))
    rv = custm.rvs(size = skill_dim).astype('float64')

    return rv

def evaluate(skill_dynamics, actor, num_episodes, env):
    ret_reward = []
    ret_length = []
    #ret_success_rate = []
    #ret_out_rate = []
    #ret_crash_vehicle_rate = []
    episode_count = 0
    planning_horizon = 1 #H_P
    primitive_holding = 25 #H_Z
    #first_skill = sample_skills(skill_dim)
    no_traj_samples = 9  #K
    available_skills = T.tensor([[1.0, 1.0],
                                [1.0, 0.0],
                                [1.0, -1.0],
                                [0.0, 1.0],
                                [0.0, 0.0],
                                [0.0, -1.0],
                                [-1.0, 1.0],
                                [-1.0, 0.0],
                                [-1.0, -1.0]], dtype=T.float, device=device_1)

    success_rate = 0
    out_rate = 0
    crash_vehicle_rate = 0
    while episode_count < num_episodes:
        obs = env.reset()
        done = False
        #########################################
        # Code for skill trajectories
        # diff_trajs = []
        # for _ in range(0, no_traj_samples):
        #     traj_skills = []
        #     for i in range(0, planning_horizon):
        #         if i==0:
        #             traj_skills.append(available_skills[_])
        #         else:
        #             traj_skills.append(sample_skills(skill_dim=2).tolist())
        #     diff_trajs.append(traj_skills)
        # diff_trajs = T.tensor(np.array(diff_trajs), dtype=T.float)
        # print(diff_trajs)
        # print(type(diff_trajs[0]))
        ########################################################
        episode_reward = 0
        episode_length = 0
        while done != True:
            velocity = []
            #cost = 0  #cost for SafeMetaDrive
            obs = T.tensor(obs, dtype=T.float, device=skill_dynamics.device)
            print(obs)
            costs = []
            for skill in available_skills:
                _, _, predicted_next_state, predicted_cost = skill_dynamics.forward(obs, skill)
                costs.append(predicted_cost)
            chosen_skill_index = costs.index(max(costs))
            chosen_skill = available_skills[chosen_skill_index]
            for j in range(0, primitive_holding):
                action, _ = actor.sample_normal(obs, chosen_skill)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length = episode_length+1
                velocity.append(info["velocity"])
                if info["arrive_dest"]:
                    # ret_success_rate.append(1)
                    success_rate += 1
                if info["out_of_road"]:
                    # ret_out_rate.append(1)
                    out_rate += 1
                if info["crash_vehicle"]:
                    # ret_crash_vehicle_rate.append(1)
                    crash_vehicle_rate += 1    
                if done:
                    break

        episode_count += 1
        ret_reward.append(episode_reward)
        ret_length.append(episode_length)

    ret = dict(
        reward = np.mean(ret_reward),
        length = np.mean(ret_length),
        # success_rate = np.mean(ret_success_rate),
        # out_rate = np.mean(ret_out_rate),
        # crash_vehicle_rate = np.mean(ret_crash_vehicle_rate),
        success_rate = success_rate / num_episodes,
        out_rate = out_rate / num_episodes,
        crash_vehicle_rate = crash_vehicle_rate / num_episodes,
    )

    return ret

if __name__ == "__main__":
    skill_dynamics= SkillDynamics()
    skill_dynamics.load_state_dict(T.load('/home/airl-gpu4/Jayabrata/Unsupervised_skills_metadrive/metadrive_dads_sd_no_encoder/models/dads_metadrive/PPO_LogIntriRew_L500/PPOskill_dynamics_eps0.2_epc30_L500.pt'))
    #print(skill_dynamics)
    skill_dynamics = skill_dynamics.eval()
    actor = ActorNetwork()
    actor.load_state_dict(T.load('/home/airl-gpu4/Jayabrata/Unsupervised_skills_metadrive/metadrive_dads_sd_no_encoder/models/dads_metadrive/PPO_LogIntriRew_L500/PPOactor_eps0.2_epc30_L500.pt'))
    #print(actor)
    actor = actor.eval()

    env = MetaDriveEnv(dict(
        # controller="joystick",
        # use_render=True,
        # manual_control=True,
        traffic_density= 0.1,
        random_traffic = False, 
        environment_num=200,
        start_seed=0,
        random_lane_width=False,#True,
        random_lane_num=False, #True,
        random_agent_model=True,
        map=3,#7
    ))

    num_episodes = 20

    ret = evaluate(skill_dynamics, actor, num_episodes, env)
    print("Evaluation result: {}".format(ret))
    