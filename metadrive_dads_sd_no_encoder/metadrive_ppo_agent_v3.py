# proximal policy optimization agent for dads (modified to accept skills as input)

import os
import re
from matplotlib import image
import torch as T
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import Transform
from torch.utils.data import TensorDataset, DataLoader
from metadrive_buffer_v3 import RLBuffer
from metadrive_networks_v3 import ActorNetwork, ValueNetwork

device_ids = [2,3]
device_1 = f'cuda:{device_ids[0]}'
device_2 = f'cuda:{device_ids[1]}'
#T.cuda.empty_cache()

def sample_action(skills):
    #lateral action
    if skills[0] == -1.0:
        steer = np.random.uniform(-1.0, -0.2, (1))
    elif skills[0] == 0.0:
        steer = np.random.uniform(-0.2, 0.2, (1))
    else:
        steer = np.random.uniform(0.2, 1.0, (1))
    #longitudinal action
    if skills[1] == -1.0:
        acc = np.random.uniform(-1.0, -0.2, (1))
    elif skills[1] == 0.0:
        acc = np.random.uniform(-0.2, 0.2, (1))
    else:
        acc = np.random.uniform(0.2, 1.0, (1))
    
    action = np.concatenate((steer, acc))

    return action

class ActorCritic(nn.Module):
    def __init__(self, 
                lr,
                obs_dims, 
                n_actions, 
                features_dim, 
                skill_dims, 
                actor_layers=6, 
                critic_layers=6):
        super(ActorCritic, self).__init__()
        
        # actor
        self.actor = ActorNetwork(lr, obs_dims, n_actions, skill_dims, actor_layers, features_dim)
        # critic
        self.critic = ValueNetwork(lr, obs_dims, n_actions, skill_dims, critic_layers, features_dim)

    def forward(self):
        raise NotImplementedError
    
    def act(self, observation, skill):
        action, action_logprob = self.actor.sample_normal(observation, skill, reparameterize=True)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, observation, skill, action):
        mu, sigma = self.actor.forward(observation, skill)
        dist = Normal(mu, sigma)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic.forward(observation, skill)

        return action_logprobs, state_values, dist_entropy

class PPOAgent():
    def __init__(self, 
                 env, 
                 lr,
                 obs_dims,
                 features_dim, 
                 n_actions, 
                 K_epochs=3,
                 skill_dims=2, 
                 gamma=0.99, 
                 max_size=10000, #1000
                 actor_layers= 6,
                 critic_layers= 6,
                 batch_size=128,  ##64
                 start_after=10000,
                 update_after=1000, 
                 chkpt_dir="models", 
                 name="dads_metadrive"):  
        self.env = env
        #self.lr = lr
        self.obs_dims = obs_dims
        self.skill_dims = skill_dims
        self.features_dim = features_dim
        self.actor_layers = actor_layers
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.K_epochs = K_epochs
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        os.makedirs(self.checkpoint_file, exist_ok=True)

        self.buffer = RLBuffer(max_size, obs_dims, n_actions, skill_dims)
        self.policy = ActorCritic(lr, obs_dims, n_actions, features_dim, skill_dims, actor_layers, critic_layers).to(device_1)
        self.optimizer = T.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr},
                        {'params': self.policy.critic.parameters(), 'lr': lr}
                    ])
        self.policy_old = ActorCritic(lr, obs_dims, n_actions, features_dim, skill_dims, actor_layers, critic_layers).to(device_1)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, observation, skill):
        with T.no_grad():
            action, action_logprob = self.policy_old.act(self.lr, self.obs_dims, self.n_actions, self.skill_dims, self.actor_layers, self.features_dim)

        return action, action_logprob
    
    def remember(self, state, skill, action, action_logprob, reward, new_state, done):
        self.buffer.store_transition(state, skill, action, action_logprob, reward, new_state, done)

    def learn(self):
        if self.step_counter < self.update_after:
            return

        # Monte Carlo estimate of returns
        state_returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.reward_memory), reversed(self.buffer.terminal_memory)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            state_returns.insert(0, discounted_reward)
        state_returns = np.array(state_returns)
        self.buffer.store_returns(state_returns)

        train_dataloader = DataLoader(dataset=self.buffer, batch_size=128)
        # Optimize policy 
        for _ in range(self.K_epochs):
            for old_states, old_skills, old_actions, old_logprobs, old_state_returns, _, _ in train_dataloader:
                # Evaluate old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_skills, old_actions)
            
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = T.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                advantages = old_state_returns - state_values.detach()   
                surr1 = ratios * advantages
                surr2 = T.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                total_loss = -T.min(surr1, surr2) + 0.5*self.MseLoss(state_values, old_state_returns) - 0.01*dist_entropy
            
                # take gradient step
                self.optimizer.zero_grad()
                total_loss.mean().backward()
                self.optimizer.step()    
                self.policy_loss = -T.min(surr1, surr2)
                self.critic_loss = 0.5*self.MseLoss(state_values, old_state_returns)
                self.dist_entropy = dist_entropy

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def get_stats(self):
        return self.policy_loss.detach().item(), self.critic_loss.detach().item(), self.dist_entropy.detach().item()

    def save_models(self, ep=0, best=True):
        if best:
            print("Saving best model")
            T.save(self.policy.actor, self.checkpoint_file+"/actor")
            T.save(self.policy.critic, self.checkpoint_file+"/critic")
        else:
            print("Saving regular model")
            T.save(self.policy.actor, self.checkpoint_file+"/actor_"+str(ep))
            T.save(self.policy.critic, self.checkpoint_file+"/critic_"+str(ep))

    def load_models(self, ep=0, best=True):
        if best:
            print("Loading best model")
            self.actor = T.load(self.checkpoint_file+"/actor")
            self.critic = T.load(self.checkpoint_file+"/critic")
        else:
            print("Loading regular model")
            self.actor = T.load(self.checkpoint_file+"/actor_"+str(ep))
            self.critic = T.load(self.checkpoint_file+"/critic_"+str(ep))


# class OLdSACAgent():
#     def __init__(self, 
#                  env, 
#                  lr, 
#                  obs_dims,
#                  features_dim, 
#                  tau, 
#                  n_actions, 
#                  K_epochs,
#                  eps_clip,
#                  action_std_init=0.6,
#                  latent_dims=2, 
#                  gamma=0.99, 
#                  max_size=10000, #1000
#                  actor_layers= 6,
#                  critic_layers= 6,
#                  batch_size=128,  ##64
#                  start_after=10000,
#                  update_after=1000, 
#                  chkpt_dir="models", 
#                  name="car"):
#         self.env = env
#         self.lr = lr
#         self.obs_dims = obs_dims
#         self.latent_dims = latent_dims
#         self.tau = tau
#         self.n_actions = n_actions
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.name = name
#         self.action_std_init = action_std_init
#         self.eps_clip = eps_clip
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
#         os.makedirs(self.checkpoint_file, exist_ok=True)

#         self.memory = RLBuffer(max_size, obs_dims, n_actions, latent_dims)

#         self.actor = ActorNetwork(lr, obs_dims, n_actions, latent_dims, actor_layers, features_dim)

#         self.value = ValueNetwork(lr, obs_dims, n_actions, latent_dims, critic_layers, features_dim)

#         self.step_counter = 0
#         self.start_after = start_after
#         self.update_after = update_after

#     def choose_action(self, observation, skill):
#         rand = np.random.random()
#         if self.step_counter < self.start_after and rand < 0.001:
#             # actions = self.env.action_space.sample()[0]
#             actions = sample_action(skill)
#             return actions
#         else:
#             state = T.tensor([observation], dtype=T.float, device=self.actor.device)
#             skill = T.tensor([skill], dtype=T.float, device=self.actor.device)
#             actions, action_logprobs = self.actor.sample_normal(state, skill, reparameterize=False)
#             #return actions.detach().cpu().numpy()[0]
#             return actions.detach(), action_logprobs.detach()

#     def remember(self, state, skill, action, reward, new_state, done):
#         self.memory.store_transition(state, skill, action, reward, new_state, done)

#     def learn(self):
#         if self.step_counter < self.update_after:
#             return

#         state, action, reward, new_state, done, skills = self.memory.sample_buffer(self.batch_size)

#         reward = T.unsqueeze(T.tensor(reward, dtype=T.float),1).to(self.critic.device)
#         done = T.unsqueeze(T.from_numpy(done).float(),1).to(self.critic.device)
#         next_state = T.tensor(new_state, dtype=T.float, device=self.critic.device)
#         state = T.tensor(state, dtype=T.float, device=self.critic.device)
#         action = T.tensor(action, dtype=T.float, device=self.critic.device)
#         skills = T.tensor(skills, dtype=T.float, device=self.critic.device)

#         with T.no_grad():
#             next_action, logprobs_next_action = self.actor.sample_normal(next_state, skills)
#             q_t1, q_t2 = self.target.forward(next_state, next_action, skills)
#             q_target = T.min(q_t1, q_t2)
#             critic_target = reward + (1.0 - done) * self.gamma * (q_target - self.alpha * logprobs_next_action)

#         q_1, q_2 = self.critic.forward(state, action, skills)
#         #print("q_1 dim: ", q_1.size())
#         #print("q_2 dim: ", q_2.size())
#         #print("critic_target dim: ", critic_target.size())
#         loss_1 = T.nn.MSELoss()(q_1, critic_target)
#         loss_2 = T.nn.MSELoss()(q_2, critic_target)
#         #print("loss_1 dim: ", loss_1.size())

#         q_loss_step = 0.5*(loss_1 + loss_2)
#         self.critic.optimizer.zero_grad()
#         q_loss_step.backward()
#         # T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.25)
#         self.critic.optimizer.step()

#         self.q1_loss = loss_1.detach().item()
#         self.q2_loss = loss_2.detach().item()

#         for p in self.critic.parameters():
#             p.requires_grad = False

#         policy_action, logprobs_policy_action = self.actor.sample_normal(state, skills)
#         p1, p2 = self.critic.forward(state, policy_action, skills)

#         target = T.min(p1, p2)
#         policy_loss = (self.alpha * logprobs_policy_action - target).mean()
#         # print("logprobs*alpha: ", self.alpha*logprobs_policy_action)
#         # print("target dimension: ",target.size())
#         # print("target: ", target)
#         self.actor.optimizer.zero_grad()
#         policy_loss.backward()
#         # T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.25)
#         self.actor.optimizer.step()
#         self.policy_loss = policy_loss.detach().item()

#         for p in self.critic.parameters():
#             p.requires_grad = True

#     def get_stats(self):
#         return self.q1_loss+self.q2_loss, self.policy_loss, self.alpha.item()

#     def save_models(self, ep=0, best=True):
#         if best:
#             print("Saving best model")
#             T.save(self.actor, self.checkpoint_file+"/actor")
#             T.save(self.critic, self.checkpoint_file+"/critic")
#             T.save(self.target, self.checkpoint_file+"/target")
#             T.save(self.log_alpha, self.checkpoint_file+"/logalpha")
#         else:
#             print("Saving regular model")
#             T.save(self.actor, self.checkpoint_file+"/actor_"+str(ep))
#             T.save(self.critic, self.checkpoint_file+"/critic_"+str(ep))
#             T.save(self.target, self.checkpoint_file+"/target_"+str(ep))
#             T.save(self.log_alpha, self.checkpoint_file+"/logalpha_"+str(ep))


#     def load_models(self, ep=0, best=True):
#         if best:
#             print("Loading best model")
#             self.actor = T.load(self.checkpoint_file+"/actor")
#             self.critic = T.load(self.checkpoint_file+"/critic")
#             self.target = T.load(self.checkpoint_file+"/target")
#             self.log_alpha = T.load(self.checkpoint_file+"/logalpha")
#         else:
#             print("Loading regular model")
#             self.actor = T.load(self.checkpoint_file+"/actor_"+str(ep))
#             self.critic = T.load(self.checkpoint_file+"/critic_"+str(ep))
#             self.target = T.load(self.checkpoint_file+"/target_"+str(ep))
#             self.log_alpha = T.load(self.checkpoint_file+"/logalpha_"+str(ep))