# Replay buffers for SAC and dads algorithm

import numpy as np
import torch as T
from torch.utils.data import Dataset

device_ids = [2,3]
device_1 = f'cuda:{device_ids[0]}'
device_2 = f'cuda:{device_ids[1]}'
T.cuda.empty_cache()

# Replay buffer for sac
class RLBuffer():
    def __init__(self, max_size, obs_dims, n_actions, latent_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, obs_dims))
        self.new_state_memory = np.zeros((self.mem_size, obs_dims))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.skill_memory = np.zeros((self.mem_size, latent_dims))

    def store_transition(self, state, skill, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.skill_memory[index] = skill

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        skills = self.skill_memory[batch]

        return states, actions, rewards, states_, dones, skills


# Replay buffer for training the skill dynamics

class DadsBuffer(Dataset):
    def __init__(self):
        self.observation_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_observation_memory = []
        self.terminal_memory = []
        self.latent_memory = []
        #self.state_delta_memory = []

    def store_transition(self, observation, action, reward, observation_, done, skill):
        self.observation_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.next_observation_memory.append(observation_)
        self.terminal_memory.append(done)
        self.latent_memory.append(skill)
        #self.state_delta_memory.append(state_delta)

    def sample_buffer(self):
        observations = np.array(self.observation_memory)
        actions = np.array(self.action_memory)
        next_observations = np.array(self.next_observation_memory)
        dones = np.array(self.terminal_memory, dtype=np.bool)
        skills = np.array(self.latent_memory)
        env_rewards = np.array(self.reward_memory)
        #state_delta = np.array(self.state_delta_memory)

        #return states, skills, state_delta, actions, next_states, dones
        return observations, skills, actions, next_observations, env_rewards, dones

    def clear_buffer(self):
        self.observation_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_observation_memory = []
        self.terminal_memory = []
        self.latent_memory = []
        #self.state_delta_memory = []

    def __getitem__(self, index):
        observations = T.tensor(self.observation_memory[index], dtype=T.float, device=device_2) #device=T.device("cuda:0")
        skills = T.tensor(self.latent_memory[index], dtype=T.float, device=device_2)
        #state_delta = T.tensor(self.state_delta_memory[index], dtype=T.float, device=device_2)
        next_observations = T.tensor(self.next_observation_memory[index], dtype=T.float, device=device_2)
        env_rewards = T.tensor(self.reward_memory[index], dtype=T.float, device = device_2)

        return observations, skills, next_observations, env_rewards

    def __len__(self):
        return len(self.observation_memory)
