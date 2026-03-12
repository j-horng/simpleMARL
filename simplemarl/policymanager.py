import numpy as np
import torch
from buffer import Buffer, DummyBuffer
from simplemarl.algorithms.ppo import PPO, Critic, Actor

class PolicyManager:
    def __init__(self,):
        self.obs_spaces = {} 
        self.act_spaces = {}
        self.policies = {}
        self.to_train = []
        self.buffers = {}
        #Setup Policies
    def add_policy(self, agent_id, obs_space, act_space, policy, to_train=False):
        self.obs_spaces[agent_id] = obs_space 
        self.act_spaces[agent_id] = act_space
        if isinstance(policy, str):
            self.policies[agent_id] = policy
        elif isinstance(policy, PPO):
            self.policies[agent_id] = policy()
        elif isinstance(policy, Actor):
            self.policies[agent_id] = policy()
        else:
            raise NotImplementedError("Currently only supports PPO, and selfplay (policy = agent_id string)")
        #TODO Add DQN, DDQN or other algorithm support 

        if to_train:
            self.buffers[agent_id] = Buffer()
        else:
            self.buffers[agent_id] = DummyBuffer()
        self.to_train.append(to_train)
        
    def add_obs(self, agent_id:str, data:[np.ndarray, torch.Tensor]):
        self.buffers[agent_id].add_obs(data)
    def add_action(self, agent_id:str, data:[np.ndarray, torch.Tensor]):
        self.buffers[agent_id].add_action(data)
    def add_value(self, agent_id:str, data:[np.ndarray, torch.Tensor]):
        if agent_id in self.to_train:
            self.buffers[agent_id].add_value(data)
    def add_reward(self,  agent_id:str, data:[np.ndarray, torch.Tensor]):
        if agent_id in self.to_train:
            self.buffers[agent_id].add_reward(data)
    def add_done(self,  agent_id:str, data:[np.ndarray, torch.Tensor]):
        if agent_id in self.to_train:
            self.buffers[agent_id].add_done(data)
    
    #Add getters
    def update(self):
        pass
    def compute_action(self, agent_id,):
        pass


class MAPPO(PolicyManager):
    def __init__(self, state_space):
        super.__init__()
        self.state_space = state_space
        self.critic = Critic(self.state_space)


    def update_critic(self,):
        pass
    def update(self, agent_id):
        pass
    def compute_action(self, agent_id, ):
        pass