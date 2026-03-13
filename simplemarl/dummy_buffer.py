import numpy as np
import torch
from gymnasium.spaces import Box
#Dummy Buffer (Holds only 1 step)
class DummyBuffer:
    def __init__(self, obs_space, act_space, num_envs, device="cpu"):
        self.num_envs = num_envs
        self.num_steps = 1 
        self.cur_step = 0
        self.obs_space = obs_space 
        self.act_space = act_space 
        self.device = device

        #Initialize Buffer
        self.observations = torch.zeros((self.num_steps, num_envs, *obs_space.shape), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((self.num_steps, num_envs, *act_space.shape), dtype=torch.float32).to(self.device)
    def full(self,):
        return False
    def get_step(self,):
        return self.cur_step
    def step(self,):
        self.cur_step = 0 #Never update 
    def reset(self):
        #Start overwriting the old values
        self.cur_step = 0
    def get_values(self):
        raise NotImplementedError("DummyBuffer only stores one step values; You are looking for Buffer class")
    def get_returns(self):
        raise NotImplementedError("DummyBuffer only stores one step values; You are looking for Buffer class")
    def get_rewards(self):
        raise NotImplementedError("DummyBuffer only stores one step values; You are looking for Buffer class")
    def get_flat_batch(self,):
        raise NotImplementedError("DummyBuffer only stores one step values; You are looking for Buffer class")
    def get_average_return(self,):
        raise NotImplementedError("DummyBuffer only stores one step values; You are looking for Buffer class")
    def calculate_returns_and_advantages(self, gamma, gae_lambda):
        raise NotImplementedError("DummyBuffer only stores one step values; You are looking for Buffer class")
