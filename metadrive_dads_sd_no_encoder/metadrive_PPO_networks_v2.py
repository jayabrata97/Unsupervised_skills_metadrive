# Defines the neural networks for the actor critics and the skill dynamics

import os
import math
from turtle import forward, st
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from torch.distributions import constraints
import torch.optim as optim
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import Transform
import copy

device_ids = [2,3]
device_1 = f'cuda:{device_ids[0]}'
device_2 = f'cuda:{device_ids[1]}'
#T.cuda.empty_cache()

# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))


class ActorNetwork(nn.Module):
    def __init__(self, 
                 lr=3e-4, #3e-7
                 obs_dims = 261,
                 action_dims = 2, 
                 skill_dims = 2, 
                 fc1_dims = 6, 
                 features_dim = 18):
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.skill_dims = skill_dims
        self.features_dim = features_dim

        self.en_linear_1 = nn.Linear(in_features = self.obs_dims, out_features=100)
        nn.init.xavier_uniform_(self.en_linear_1.weight.data)
        self.en_linear_2 = nn.Linear(in_features=100, out_features = self.features_dim)
        nn.init.xavier_uniform_(self.en_linear_2.weight.data)
       
        self.fc1 = nn.Linear(in_features = self.features_dim + self.skill_dims, out_features = fc1_dims)
        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.mu = nn.Linear(in_features = fc1_dims, out_features = self.action_dims)
        # self.logsigma = nn.Linear(in_features = fc1_dims, out_features = self.action_dims)
        self.sigma = nn.Linear(in_features = fc1_dims, out_features = self.action_dims)
        nn.init.xavier_uniform_(self.mu.weight.data)
        # nn.init.xavier_uniform_(self.logsigma.weight.data)
        nn.init.xavier_uniform_(self.sigma.weight.data)

        #self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device(device_1 if T.cuda.is_available() else 'cpu')
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, skill):
        state = self.en_linear_1(observation)
        state = F.relu(state)
        encoded_state = self.en_linear_2(state)
        x = T.cat((encoded_state, skill), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.mu(x)
        # sigma = self.sigma(x)
        # sigma = T.abs(sigma)  ##for preventing the negative values of std
        logsigma = self.sigma(x)
        sigma = logsigma.exp()
        return mu, sigma

    def sample_normal(self, observation, skill, reparameterize=True):
        mu, sigma = self.forward(observation, skill)
        # logsigma = T.clamp(logsigma, -20, 2)
        # sigma = logsigma.exp()
        sigma = T.clamp(sigma, min=0.3, max=2)
        probabilities = Normal(mu, sigma)
        transforms = [TanhTransform(cache_size=1)]
        probabilities = TransformedDistribution(probabilities, transforms)
        if reparameterize:
            action = probabilities.rsample()
        else:
            action = probabilities.sample()

        log_probs = probabilities.log_prob(action).sum(axis=-1, keepdim=True)
        log_probs.to(self.device)
        
        return action, log_probs, mu, sigma

class ValueNetwork(nn.Module):
    def __init__(self,
                 lr=1e-3,#1e-4,3e-4,
                 obs_dims=261,
                 action_dims=2,
                 skill_dims=2,
                 fc1_dims=6,
                 features_dim=18):
        super(ValueNetwork, self).__init__()
        self.lr = lr
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.skill_dims = skill_dims
        self.features_dim = features_dim
        self.en_linear_1 = nn.Linear(in_features = self.obs_dims, out_features = 100)
        nn.init.xavier_uniform_(self.en_linear_1.weight.data)
        self.en_linear_2 = nn.Linear(in_features = 100, out_features = features_dim)
        nn.init.xavier_uniform_(self.en_linear_2.weight.data)

        self.fc1 = nn.Linear(in_features = self.features_dim + self.skill_dims, out_features=fc1_dims)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.v = nn.Linear(in_features=fc1_dims, out_features=1)
        nn.init.xavier_uniform_(self.v.weight.data)

        #self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device(device_1 if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, skill):
        state = self.en_linear_1(observation)
        state = F.relu(state)
        encoded_state = self.en_linear_2(state)
        x = T.cat((encoded_state, skill), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        v = self.v(x)

        return v

# TODO: Does the critic evaluate the skill? Yes
class CriticNetwork(nn.Module):
    def __init__(self, 
                 lr=3e-4,#3e-7
                 obs_dims = 261,
                 action_dims=2, 
                 skill_dims=2, 
                 fc1_dims=6, 
                 features_dim=18):
        super(CriticNetwork, self).__init__()
        self.lr = lr
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.skill_dims = skill_dims
        self.features_dim = features_dim

        self.en_linear_1 = nn.Linear(in_features = self.obs_dims, out_features=100)
        nn.init.xavier_uniform_(self.en_linear_1.weight.data)
        self.en_linear_2 = nn.Linear(in_features=100, out_features = self.features_dim)
        nn.init.xavier_uniform_(self.en_linear_2.weight.data)

        self.fc1 = nn.Linear(in_features = self.features_dim + self.action_dims + self.skill_dims, out_features=fc1_dims)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.q = nn.Linear(in_features=fc1_dims, out_features=1)
        nn.init.xavier_uniform_(self.q.weight.data)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device(device_1 if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, action, skill):
        state = self.en_linear_1(observation)
        state = F.relu(state)
        encoded_state = self.en_linear_2(state)
        x = T.cat((encoded_state, action, skill), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        q = self.q(x)

        return q

class DoubleCriticNetwork(nn.Module):
    def __init__(self, 
                 lr=3e-4,#3e-7
                 obs_dims = 261,
                 action_dims=2, 
                 skill_dims=2, 
                 fc1_dims=6, 
                 features_dim=18):
        super(DoubleCriticNetwork, self).__init__()
        self.net1 = CriticNetwork(lr=lr,
                                  obs_dims=obs_dims,
                                  action_dims=action_dims,
                                  skill_dims=skill_dims,
                                  fc1_dims=fc1_dims,
                                  features_dim=features_dim)
        
        self.net2 = CriticNetwork(lr=lr,
                                  obs_dims=obs_dims,
                                  action_dims=action_dims,
                                  skill_dims=skill_dims,
                                  fc1_dims=fc1_dims,
                                  features_dim=features_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device(device_1 if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, action, skill):
        return self.net1(observation, action, skill), self.net2(observation, action, skill)


# TODO: Implement mixture of experts
# TODO: Should we implement normalization of the input observation
class SkillDynamics(nn.Module):
    def __init__(self,
                 lr=3e-4,#3e-7
                 obs_dims = 261,
                 skill_dims=2,
                 fc1_dims=6,
                 features_dim=18):
        super(SkillDynamics, self).__init__()

        self.lr = lr
        self.skill_dims = skill_dims
        self.features_dim = features_dim
        self.obs_dims = obs_dims

        self.en_linear_1 = nn.Linear(in_features = self.obs_dims, out_features=100)
        nn.init.xavier_uniform_(self.en_linear_1.weight.data)
        #self.en_linear_1_bn = nn.BatchNorm1d(100)
        self.en_linear_2= nn.Linear(in_features=100, out_features = self.features_dim)
        nn.init.xavier_uniform_(self.en_linear_2.weight.data)
        #self.en_linear_2_bn = nn.BatchNorm1d(self.features_dim)
        self.de_mean = nn.Linear(in_features = self.features_dim + self.skill_dims, out_features = self.features_dim)
        nn.init.xavier_uniform_(self.de_mean.weight.data)
        #self.de_mean_bn = nn.BatchNorm1d(self.features_dim)
        self.de_logsigma = nn.Linear(in_features = self.features_dim + self.skill_dims, out_features = self.features_dim)
        nn.init.xavier_uniform_(self.de_logsigma.weight.data)
        #self.de_logsigma_bn = nn.BatchNorm1d(self.features_dim)

        self.fc1 = nn.Linear(in_features = self.obs_dims, out_features=100)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        #self.fc1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(in_features=100, out_features = self.features_dim)
        nn.init.xavier_uniform_(self.fc2.weight.data)

        self.recons_linear_1 = nn.Linear(in_features = self.obs_dims, out_features=100)
        self.recons_linear_2= nn.Linear(in_features=100, out_features = self.features_dim)
        self.fc3 = nn.Linear(in_features = self.features_dim, out_features=1)
        nn.init.xavier_uniform_(self.fc3.weight.data)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device(device_1 if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, skill):
        state = self.en_linear_1(observation)
        state = F.relu(state)
        encoded_state = self.en_linear_2(state)
        x = T.cat((encoded_state, skill), dim=-1)
        de_mean = self.de_mean(x)
        de_logsigma = self.de_logsigma(x)
        de_sigma = de_logsigma.exp()
        de_sigma = T.clamp(de_sigma, 0.4, 10)
        de_probs = Normal(de_mean, de_sigma)
        predicted_next_state = de_probs.rsample()
        predicted_cost = self.fc3(predicted_next_state)

        return de_mean, de_sigma, predicted_next_state, predicted_cost

    def get_log_probs(self, observation, skill, next_observation, env_reward):  
        de_mean, de_sigma, predicted_next_state, predicted_cost = self.forward(observation, skill)
        next_state = self.fc1(next_observation)
        next_state = self.fc2(next_state)
        de_probs = Normal(de_mean, de_sigma)
        log_probs = de_probs.log_prob((next_state)).sum(axis=-1, keepdim=True)
        log_probs.to(self.device)
        return log_probs
 
    def get_reconstruction_loss(self, observation, skill, next_observation, env_reward):
        de_mean, de_sigma, predicted_next_state, predicted_cost = self.forward(observation, skill)
        #print("forward called for second time")

        for p, q in zip(self.en_linear_1.parameters(), self.recons_linear_1.parameters()):
            q.data.copy_(p.data)
        for p in self.recons_linear_1.parameters():
            p.requires_grad = False
        for p, q in zip(self.en_linear_2.parameters(), self.recons_linear_2.parameters()):
            q.data.copy_(p.data)
        for p in self.recons_linear_2.parameters():
            p.requires_grad = False

        next_state = self.recons_linear_1(next_observation)
        next_state = F.relu(next_state)
        next_state = self.recons_linear_2(next_state)
        loss = nn.MSELoss()
        reconstruction_loss = loss(next_state, predicted_next_state)
        env_reward = env_reward.unsqueeze(1)
        env_reward = env_reward.float()
        cost_func_loss = loss(predicted_cost, env_reward)

        return reconstruction_loss, cost_func_loss

    ## TODO: for get_log_probs(), check if we require - (negative) in front 
    def get_loss(self, observation, skill, next_observation, env_reward):
        return -self.get_log_probs(observation, skill, next_observation, env_reward).mean(), self.get_reconstruction_loss(observation, skill, next_observation, env_reward)
        # return self.get_log_probs(observation, skill, next_observation).mean()
