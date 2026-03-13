#Code contains information needed for training a MAPPO Actor agent
import numpy as np
import torch
from gymnasium.spaces import Box

class ActorBuffer:
    def __init__(self, obs_space, act_space, num_envs, num_steps, device):
        self.num_envs = num_envs
        self.num_steps = num_steps 
        self.cur_step = 0
        self.obs_space = obs_space 
        self.act_space = act_space
        self.device = device

        self.observations = torch.zeros((self.num_steps, self.num_envs, *self.obs_space.shape), dtype=torch.float32)
        self.actions = torch.zeros((self.num_steps, self.num_envs, *act_space.shape), dtype=torch.float32)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    def full(self,):
        return self.cur_step >= self.num_steps
    def get_step(self,):
        return self.cur_step
    def step(self,):
        self.cur_step += 1
    def reset(self):
        #Start overwriting the old values
        self.cur_step = 0
    
    def get_flat_batch(self,):
        flat_obs = self.observations.reshape((-1,) + self.obs_space.shape)
        flat_actions = self.actions.reshape(-1)
        flat_logprobs = self.logprobs.reshape(-1)
        return {'logprobs':flat_logprobs, 'observations':flat_obs, 'actions':flat_actions}

    