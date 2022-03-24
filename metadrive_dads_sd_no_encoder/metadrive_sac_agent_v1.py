# soft actor critic agent for dads (modified to accept skills as input)

import os
from matplotlib import image
import torch as T
import copy
import numpy as np
import torch.nn.functional as F
from metadrive_buffer_v1 import RLBuffer
from metadrive_networks_v1 import ActorNetwork, CriticNetwork, DoubleCriticNetwork

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

class SacAgent():
    def __init__(self, 
                 env, 
                 lr, 
                 obs_dims,
                 features_dim, 
                 tau, 
                 n_actions, 
                 latent_dims=2, 
                 gamma=0.99, 
                 max_size=10000, #1000
                 actor_layers= 6,
                 critic_layers= 6,
                 batch_size=128,  ##64
                 start_after=10000,
                 update_after=1000, 
                 chkpt_dir="models", 
                 name="car"):
        self.env = env
        self.lr = lr
        self.obs_dims = obs_dims
        self.latent_dims = latent_dims
        self.tau = tau
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        os.makedirs(self.checkpoint_file, exist_ok=True)

        self.memory = RLBuffer(max_size, obs_dims, n_actions, latent_dims)

        self.actor = ActorNetwork(lr, obs_dims, n_actions, latent_dims,
                                  actor_layers, features_dim)

        self.critic = DoubleCriticNetwork(lr, obs_dims, n_actions, latent_dims, 
                                          critic_layers, features_dim)

        self.target = copy.deepcopy(self.critic)

        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
        self.alpha_optim = T.optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp()

        self.step_counter = 0
        self.start_after = start_after
        self.update_after = update_after

    def choose_action(self, observation, skill):
        rand = np.random.random()
        if self.step_counter < self.start_after and rand < 0.001:
            # actions = self.env.action_space.sample()[0]
            actions = sample_action(skill)
            return actions
        else:
            state = T.tensor([observation], dtype=T.float, device=self.actor.device)
            skill = T.tensor([skill], dtype=T.float, device=self.actor.device)
            actions, _ = self.actor.sample_normal(state, skill, reparameterize=False)
            return actions.detach().cpu().numpy()[0]

    def remember(self, state, skill, action, reward, new_state, done):
        self.memory.store_transition(state, skill, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        with T.no_grad():
            for target_params, critic_params in zip(self.target.parameters(), self.critic.parameters()):
                target_params.data.copy_((1-tau)*critic_params.data + tau*target_params.data)

    def learn(self):
        if self.step_counter < self.update_after:
            return

        state, action, reward, new_state, done, skills = self.memory.sample_buffer(self.batch_size)

        reward = T.unsqueeze(T.tensor(reward, dtype=T.float),1).to(self.critic.device)
        done = T.unsqueeze(T.from_numpy(done).float(),1).to(self.critic.device)
        next_state = T.tensor(new_state, dtype=T.float, device=self.critic.device)
        state = T.tensor(state, dtype=T.float, device=self.critic.device)
        action = T.tensor(action, dtype=T.float, device=self.critic.device)
        skills = T.tensor(skills, dtype=T.float, device=self.critic.device)

        with T.no_grad():
            next_action, logprobs_next_action = self.actor.sample_normal(next_state, skills)
            q_t1, q_t2 = self.target.forward(next_state, next_action, skills)
            q_target = T.min(q_t1, q_t2)
            critic_target = reward + (1.0 - done) * self.gamma * (q_target - self.alpha * logprobs_next_action)

        q_1, q_2 = self.critic.forward(state, action, skills)
        loss_1 = T.nn.MSELoss()(q_1, critic_target)
        loss_2 = T.nn.MSELoss()(q_2, critic_target)

        q_loss_step = loss_1 + loss_2
        self.critic.optimizer.zero_grad()
        q_loss_step.backward()
        # T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.25)
        self.critic.optimizer.step()

        self.q1_loss = loss_1.detach().item()
        self.q2_loss = loss_2.detach().item()

        for p in self.critic.parameters():
            p.requires_grad = False

        policy_action, logprobs_policy_action = self.actor.sample_normal(state, skills)
        p1, p2 = self.critic.forward(state, policy_action, skills)

        target = T.min(p1, p2)
        policy_loss = (self.alpha * logprobs_policy_action - target).mean()
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        # T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.25)
        self.actor.optimizer.step()
        self.policy_loss = policy_loss.detach().item()

        temp_loss = -self.log_alpha * (logprobs_policy_action.detach() + self.target_entropy).mean()
        self.alpha_optim.zero_grad()
        temp_loss.backward()
        # T.nn.utils.clip_grad_norm_(self.alpha_optim.parameters(), 1e3)
        self.alpha_optim.step()

        for p in self.critic.parameters():
            p.requires_grad = True
        
        self.alpha = self.log_alpha.exp()

        self.update_network_parameters()

    def get_stats(self):
        return self.q1_loss+self.q2_loss, self.policy_loss, self.alpha.item()

    def save_models(self, ep=0, best=True):
        if best:
            print("Saving best model")
            T.save(self.actor, self.checkpoint_file+"/actor")
            T.save(self.critic, self.checkpoint_file+"/critic")
            T.save(self.target, self.checkpoint_file+"/target")
            T.save(self.log_alpha, self.checkpoint_file+"/logalpha")
        else:
            print("Saving regular model")
            T.save(self.actor, self.checkpoint_file+"/actor_"+str(ep))
            T.save(self.critic, self.checkpoint_file+"/critic_"+str(ep))
            T.save(self.target, self.checkpoint_file+"/target_"+str(ep))
            T.save(self.log_alpha, self.checkpoint_file+"/logalpha_"+str(ep))


    def load_models(self, ep=0, best=True):
        if best:
            print("Loading best model")
            self.actor = T.load(self.checkpoint_file+"/actor")
            self.critic = T.load(self.checkpoint_file+"/critic")
            self.target = T.load(self.checkpoint_file+"/target")
            self.log_alpha = T.load(self.checkpoint_file+"/logalpha")
        else:
            print("Loading regular model")
            self.actor = T.load(self.checkpoint_file+"/actor_"+str(ep))
            self.critic = T.load(self.checkpoint_file+"/critic_"+str(ep))
            self.target = T.load(self.checkpoint_file+"/target_"+str(ep))
            self.log_alpha = T.load(self.checkpoint_file+"/logalpha_"+str(ep))
