# trainig loop for dads. PPO agent

#from traceback import print_tb
# import gym
import numpy as np
from scipy import stats
# from gym.wrappers import RescaleAction
from metadrive_buffer_v3 import DadsBuffer
from metadrive_networks_v3 import *
from metadrive_ppo_agent_v3 import PPOAgent
from metadrive import MetaDriveEnv
#from metadrive import SafeMetaDriveEnv
import torch as T
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
# import matplotlib.pyplot as plt
# import argparse
# import random

device_ids = [2,3]
device_1 = f'cuda:{device_ids[0]}'
device_2 = f'cuda:{device_ids[1]}'

T.cuda.empty_cache()

def sample_skills(skill_dim):
    a = np.array([-1.0, 0.0, 1.0])
    pk = ((1/3), (1/3), (1/3))
    custm = stats.rv_discrete(name='custm', values=(a, pk))
    rv = custm.rvs(size = skill_dim).astype('float64')

    return rv

def run_episode(env, agent, skill_dynamics, buffer, steps_per_episode, skill_dims, step_counter):
    obs = env.reset()
    step_counter_local = 0
    skill = sample_skills(skill_dims)
    cumulative_env_reward = 0
    while step_counter_local < steps_per_episode:
        action, action_logprob = agent.select_action(obs, skill)
        agent.step_counter += 1
        obs_, reward, done, info = env.step(action)
        step_counter_local += 1
        step_counter += 1
        cumulative_env_reward += reward

        # buffer.store_transition(obs, action, obs_, done, skill, obs_-obs)
        buffer.store_transition(obs, action, action_logprob, reward, obs_, done, skill)
        obs = obs_
        if done == True:
            obs = env.reset()
    
    return buffer, step_counter, cumulative_env_reward

# TODO: optimize this function
def compute_dads_reward(agent, skill_dynamics, dads_buffer, available_skills, step_counter):
# def compute_dads_reward(agent, skill_dynamics, dads_buffer, skill_dims, step_counter):
    L = 9 #500 #100
    observations, skills, actions, action_logprobs, next_observations, env_rewards, dones = dads_buffer.sample_buffer()
    denom_skills = available_skills
    # denom_skills = T.tensor(np.random.randint(-1, 1+1, (L, skill_dims)), dtype=T.float, device=skill_dynamics.device)

    for i in range(len(observations)):
        local_state_tensor = T.tensor(np.array([observations[i]]), dtype=T.float, device=skill_dynamics.device)
        local_skill_tensor = T.tensor(np.array([skills[i]]), dtype=T.float, device=skill_dynamics.device)
        # local_delta_state = T.tensor([state_delta[i]], dtype=T.float,device=skill_dynamics.device)
        local_next_state_tensor = T.tensor(np.array([next_observations[i]]), dtype=T.float, device=skill_dynamics.device)
        local_env_reward = T.tensor(np.array([env_rewards[i]]), dtype=T.float, device=skill_dynamics.device)

        # numerator = skill_dynamics.get_log_probs(local_state_tensor, local_skill_tensor, local_delta_state).detach().cpu().numpy()[0][0]
        numerator = skill_dynamics.get_log_probs(local_state_tensor, local_skill_tensor, local_next_state_tensor, local_env_reward).exp().detach().cpu().numpy()[0][0]
        local_state_tensor = local_state_tensor.repeat(L,1)
        # local_delta_state = local_delta_state.repeat(L,1,1,1)
        local_next_state_tensor = local_next_state_tensor.repeat(L,1)
        # denom = skill_dynamics.get_log_probs(local_state_tensor, denom_skills, local_delta_state).detach().cpu().numpy().sum()
        denom = skill_dynamics.get_log_probs(local_state_tensor, denom_skills, local_next_state_tensor, local_env_reward).exp().detach().cpu().numpy().sum()
        # if numerator == 0.0:
        #     numerator = 1e-3
        # if denom == 0.0:
        #     denom = 1e-3
        if step_counter < 2e6:
            total_reward = env_rewards[i]
        else:
            intrinsic_reward = np.log(numerator/denom) + np.log(L)
            total_reward = intrinsic_reward + env_rewards[i]
        #agent.remember(observations[i], skills[i], actions[i], intrinsic_reward, next_observations[i], dones[i])
        agent.remember(observations[i], skills[i], actions[i], action_logprobs[i], total_reward, next_observations[i], dones[i])
    dads_buffer.clear_buffer()

if __name__ == '__main__':
    writer = SummaryWriter()
    # traffic_density = np.random.uniform(0.4, 0.6)
    # print("Sampled traffic density is: ", traffic_density)
    env = MetaDriveEnv(dict(
        # controller="joystick",
        # use_render=True,
        # manual_control=True,
        traffic_density= 0.1, #0.1, currently it is sampling only once, not every 1000 steps
        random_traffic = False, 
        environment_num=1000,
        start_seed=1000,
        #start_seed=random.randint(0, 1000)
        random_lane_width=False,
        random_agent_model=True,
        random_lane_num=False,
        map=3
    ))
    n_actions = 2 
    skill_dims = 2
    n_steps = 3e6 #1e7
    steps_per_episode = 1000 #200
    M = 10
    K1 = 32

    available_skills = T.tensor([[1.0, 1.0],
                                 [1.0, 0.0],
                                 [1.0, -1.0],
                                 [0.0, 1.0],
                                 [0.0, 0.0],
                                 [0.0, -1.0],
                                 [-1.0, 1.0],
                                 [-1.0, 0.0],
                                 [-1.0, -1.0]], dtype=T.float, device=device_1)  ## trying with T.device('cuda:0')

    dads_buffer = DadsBuffer()

    agent = PPOAgent(env=env,
                     lr=1e-4,#3e-4
                     obs_dims=261,
                     features_dim=18, #512
                     n_actions=n_actions,
                     skill_dims=skill_dims,
                     chkpt_dir="models",
                     name="dads_metadrive")

    skill_dynamics = SkillDynamics(lr=3e-4,#3e-7
                                   obs_dims = 261,
                                   skill_dims=skill_dims,
                                   fc1_dims=6,
                                   features_dim=18)
    step_counter = 0
    obs = env.reset()
    done = False
    pbar = tqdm(total=n_steps)
    tqdm_count = 0

    while step_counter < n_steps:
        for _ in range(M):
            dads_buffer, step_counter, cumulative_env_reward = run_episode(env, agent, skill_dynamics, dads_buffer,  ##sim_out is extra
                                                    steps_per_episode, skill_dims, step_counter)
            print(" step counter:",agent.step_counter, end='\r')
            writer.add_scalar("Cumulative env reward", cumulative_env_reward, step_counter)

        sd_data_loader = DataLoader(dataset=dads_buffer, batch_size=128) 
        for _ in range(K1):
            for observation_batch, skill_batch, next_observation_batch, env_reward_batch in sd_data_loader:
                # loss = skill_dynamics.get_loss(state_batch, skill_batch, state_delta_batch)
                # print("\n Skill dynamics loss: ",loss.item())
                # writer.add_scalar("Skill Dynamics Loss", loss, step_counter)
                skill_dynamics_loss, (reconstruction_loss, cost_func_loss) = skill_dynamics.get_loss(observation_batch, skill_batch, next_observation_batch, env_reward_batch)
                loss = skill_dynamics_loss + reconstruction_loss + cost_func_loss
                #print(" Skill dynamics loss: ",skill_dynamics_loss.item(), end='\r')
                #print(" Observation reconstruction loss: ",reconstruction_loss.item(), end='\r')
                #print(" Cost func loss: ",cost_func_loss.item(), end='\r')
                print(" Total loss: ",loss.item(), end='\r')
                writer.add_scalar("Total loss", loss.item(), step_counter)
                writer.add_scalar("Skill Dynamics Loss", skill_dynamics_loss.item(), step_counter)
                writer.add_scalar("Observation Reconstruction loss", reconstruction_loss.item(), step_counter)
                writer.add_scalar("Cost function loss", cost_func_loss.item(), step_counter)
                skill_dynamics.optimizer.zero_grad()
                loss.backward()
                skill_dynamics.optimizer.step()

        compute_dads_reward(agent, skill_dynamics, dads_buffer, available_skills, step_counter)
        # compute_dads_reward(agent, skill_dynamics, dads_buffer, skill_dims, step_counter)
        #for _ in range(128):
        agent.learn()
        agent.save_models()
        T.save(skill_dynamics.state_dict(), './models/dads_metadrive/PPOskill_dynamics_eps0.2_epc30_L9_NewRew.pt')
        # critic_loss, policy_loss, alpha = agent.get_stats()
        policy_loss, critic_loss, dist_entropy, advantages = agent.get_stats()
        print(" Critic loss: ",critic_loss)
        print(" Policy loss: ",policy_loss)
        print(" Distribution entropy: ", dist_entropy)
        writer.add_scalar("Critic Loss", critic_loss, step_counter)
        writer.add_scalar("Policy Loss", policy_loss, step_counter)
        # writer.add_scalar("Alpha", alpha, step_counter)
        writer.add_scalar("Dist entropy", dist_entropy, step_counter)
        writer.add_scalar("Advantages", advantages, step_counter)
        pbar.update(step_counter-tqdm_count)
        tqdm_count = step_counter
    pbar.close()
