#Code contains information needed for training a MAPPO critic agent
import numpy as np
import torch
from gymnasium.spaces import Box

class CriticBuffer:
    def __init__(self, state_space, num_envs, num_steps, device):
        self.num_envs = num_envs
        self.num_steps = num_steps 
        self.cur_step = 0
        self.state_space = state_space 
        self.device = device
        

        self.states = torch.zeros((self.num_steps, self.num_envs, *self.state_space.shape), dtype=torch.float32)
        self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        self.returns = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        self.advantages = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        
        self.next_done = torch.zeros((self.num_envs, ), dtype=torch.float32)
        self.next_value = torch.zeros((self.num_envs,), dtype=torch.float32)
    def full(self,):
        return self.cur_step >= self.num_steps
    def get_step(self,):
        return self.cur_step
    def step(self,):
        self.cur_step += 1
    def reset(self):
        #Start overwriting the old values
        self.cur_step = 0
    def get_states(self):
        return self.states[:self.cur_step].reshape(-1)
    def get_values(self):
        return self.values[:self.cur_step].reshape(-1)
    def get_rewards(self):
        return self.rewards[:self.cur_step]
    def get_returns(self):
        return self.returns[:self.cur_step].reshape(-1)
    
    def get_flat_batch(self,):
        flat_states = self.states.reshape((-1,) + self.state_space.shape)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)
        
        return {'states':flat_states, 'advantages':flat_advantages, 'returns':flat_returns, 'values':flat_values}

    def calculate_returns_and_advantages(self, gamma, gae_lambda):
        with torch.no_grad():
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps-1:
                    nextnonterminal = 1.0 - self.next_done#Truncate all env ironments at the last step in buffer
                    nextvalues = self.next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t+1]
                    nextvalues = self.values[t+1]
                delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam 
            self.returns = self.advantages + self.values