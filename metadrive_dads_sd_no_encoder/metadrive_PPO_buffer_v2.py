# Replay buffers for PPO and dads algorithm

from dataclasses import replace
import numpy as np
import torch as T
from torch.utils.data import Dataset

device_ids = [2,3]
device_1 = f'cuda:{device_ids[0]}'
device_2 = f'cuda:{device_ids[1]}'
T.cuda.empty_cache()

# Replay buffer for action policy
class RLBuffer():
    def __init__(self, max_size, obs_dims, n_actions, skill_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, obs_dims))
        self.new_state_memory = np.zeros((self.mem_size, obs_dims))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.action_logprob_memory = np.zeros(self.mem_size)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.skill_memory = np.zeros((self.mem_size, skill_dims))
        self.state_return = np.zeros(self.mem_size)
        self.device = T.device(device_1 if T.cuda.is_available() else 'cpu')

    def store_transition(self, state, skill, action, action_logprob, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.action_logprob_memory[index] = action_logprob
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.skill_memory[index] = skill

        self.mem_cntr += 1

    def store_returns(self, state_returns):
        self.state_return = state_returns

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        action_logprobs = self.action_logprob_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        skills = self.skill_memory[batch]
        state_returns = self.state_return[batch]

        return states, actions, action_logprobs, rewards, states_, dones, skills, state_returns

    def clear(self, max_size, obs_dims, n_actions, skill_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, obs_dims))
        self.new_state_memory = np.zeros((self.mem_size, obs_dims))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.action_logprob_memory = np.zeros(self.mem_size)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.skill_memory = np.zeros((self.mem_size, skill_dims))
        self.state_return = np.zeros(self.mem_size)

    def __getitem__(self, index):
        states = T.tensor(self.state_memory[index], dtype=T.float, device=self.device)
        skills = T.tensor(self.skill_memory[index], dtype=T.float, device=self.device)
        actions = T.tensor(self.action_memory[index], dtype=T.float, device=self.device)
        action_logprobs = T.tensor(self.action_logprob_memory[index], dtype=T.float, device=self.device)
        state_returns = T.tensor(self.state_return[index], dtype=T.float, device=self.device)
        new_states = T.tensor(self.new_state_memory[index], dtype=T.float, device = self.device)
        dones = T.tensor(self.terminal_memory[index], dtype=T.float, device=self.device)

        return states, skills, actions, action_logprobs, state_returns, new_states, dones

    def __len__(self):
        return len(self.state_memory)

# Replay buffer for training the skill dynamics

class DadsBuffer(Dataset):
    def __init__(self):
        self.observation_memory = []
        self.action_memory = []
        self.action_logprob_memory = []
        self.reward_memory = []
        self.next_observation_memory = []
        self.terminal_memory = []
        self.skill_memory = []
        #self.state_delta_memory = []
        self.device = T.device(device_1 if T.cuda.is_available() else 'cpu')

    def store_transition(self, observation, action, action_logprob, reward, observation_, done, skill):
        self.observation_memory.append(observation)
        self.action_memory.append(action)
        self.action_logprob_memory.append(action_logprob)
        self.reward_memory.append(reward)
        self.next_observation_memory.append(observation_)
        self.terminal_memory.append(done)
        self.skill_memory.append(skill)
        #self.state_delta_memory.append(state_delta)

    def sample_buffer(self):
        observations = np.array(self.observation_memory)
        actions = np.array(self.action_memory)
        action_logprobs = np.array(self.action_logprob_memory)
        next_observations = np.array(self.next_observation_memory)
        dones = np.array(self.terminal_memory, dtype=np.bool)
        skills = np.array(self.skill_memory)
        env_rewards = np.array(self.reward_memory)
        #state_delta = np.array(self.state_delta_memory)

        #return states, skills, state_delta, actions, next_states, dones
        return observations, skills, actions, action_logprobs, next_observations, env_rewards, dones

    def clear_buffer(self):
        self.observation_memory = []
        self.action_memory = []
        self.action_logprob_memory = []
        self.reward_memory = []
        self.next_observation_memory = []
        self.terminal_memory = []
        self.skill_memory = []
        #self.state_delta_memory = []

    def __getitem__(self, index):
        observations = T.tensor(self.observation_memory[index], dtype=T.float, device=self.device) #device=T.device("cuda:0")
        skills = T.tensor(self.skill_memory[index], dtype=T.float, device=self.device)
        #state_delta = T.tensor(self.state_delta_memory[index], dtype=T.float, device=device_1)
        next_observations = T.tensor(self.next_observation_memory[index], dtype=T.float, device=self.device)
        env_rewards = T.tensor(self.reward_memory[index], dtype=T.float, device = self.device)

        return observations, skills, next_observations, env_rewards

    def __len__(self):
        return len(self.observation_memory)
