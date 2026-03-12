import numpy as np
import torch
from buffer import Buffer, DummyBuffer, StateBuffer, ActorBuffer
from simplemarl.algorithms.ppo import PPO, Critic, Actor

class PolicyManager:
    def __init__(self, num_envs, num_steps, device):
        self.num_envs = num_envs 
        self.num_steps = num_steps 
        self.device = device
        self.obs_spaces = {} 
        self.act_spaces = {}
        self.policies = {}
        self.to_train = []
        self.buffers = {}

        #Setup Policies
    def add_policy(self, agent_id, obs_space, act_space, policy, to_train=False, config=None):
        assert (agent_id not in self.buffers), "Currently you add two policies for one agent"
        self.obs_spaces[agent_id] = obs_space 
        self.act_spaces[agent_id] = act_space
        if not to_train:
            self.buffers[agent_id] = DummyBuffer(obs_space, act_space, self.num_envs, self.num_steps, self.device) 
        else:
            self.to_train.append(agent_id)
            
        if isinstance(policy, str):
            self.policies[agent_id] = policy
        elif isinstance(policy, PPO):
            self.policies[agent_id] = policy(obs_space, act_space, config)
            self.buffers[agent_id] = Buffer(obs_space, act_space, self.num_envs, self.num_steps, self.device)
        # elif isinstance(policy, Actor):
        #     self.policies[agent_id] = policy(obs_space, act_space, config)
        #     self.buffers[agent_id] = ActorBuffer(obs_space, act_space, self.num_envs, self.num_steps, self.device)
        else:
            raise NotImplementedError("Currently only supports PPO, and selfplay (policy = agent_id string)")
        #TODO: Check and load a policy if passed in
        #TODO Add DQN, DDQN or other algorithm support 
    def add_obs(self, agent_id:str, data:[np.ndarray, torch.Tensor]):
        self.buffers[agent_id].add_obs(data)
    def add_action(self, agent_id:str, data:[np.ndarray, torch.Tensor]):
        self.buffers[agent_id].add_action(data)
    def add_reward(self,  agent_id:str, data:[np.ndarray, torch.Tensor]):
        if agent_id in self.to_train:
            self.buffers[agent_id].add_reward(data)
    def add_done(self,  agent_id:str, data:[np.ndarray, torch.Tensor]):
        if agent_id in self.to_train:
            self.buffers[agent_id].add_done(data)
    #Not supported in default policy manager
    def add_state(self):
        return 
    #TODO: In update maybe just pass in mb_inds?
    def update(self, batch_size, minibatch_size, update_epochs):
        #Setup flat batches
        logs = {aid:{} for aid in self.to_train}
        flat_batches = {}
        flat_batches[aid] = {self.buffers[aid].get_flat_batch() for aid in self.to_train}
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size 
                mb_inds = b_inds[start:end]
                for aid in self.to_train:
                    minibatch = {k: v[mb_inds] for k, v in flat_batches[aid].items()}
                    logs[aid] = self.policies[aid].update(minibatch)
        return logs
    def compute_action(self, agent_id, obs):
        action, logprob, _, value = self.policies[agent_id].get_action_and_value(obs)
        self.buffers[agent_id].add_value(value)
        self.buffers[agent_id].add_logprob(logprob)
        self.buffers[agent_id].add_action(action)
        #TODO: Check return type
        return action
    
    #Buffer Updates
    def reset_buffers(self):
        for aid in self.to_train:
            self.buffers[aid].reset()
    def step_buffers(self):
        for aid in self.to_train:
            self.buffers[aid].step()
    def step_buffer(self, agent_id):
        self.buffers[agent_id].step()


class MAPPO(PolicyManager):
    def __init__(self, state_space, num_envs, num_steps, device):
        super.__init__(num_envs, num_steps, device)
        self.state_space = state_space

        self.critic = Critic(self.state_space)
        self.state_buffer = StateBuffer(state_space)

    def add_state(self, state):
        _,_,_,value = self.critic(state)
        self.state_buffer.add_state(state)
        self.state_buffer.add_value(value)
    
    def add_reward(self,  agent_id:str, data:[np.ndarray, torch.Tensor]):
        if agent_id in self.to_train:
            self.buffers[agent_id].add_reward(data)
    def add_done(self,  agent_id:str, data:[np.ndarray, torch.Tensor]):
        self.buffers[agent_id].add_done(data)

    def update(self, batch_size, minibatch_size, update_epochs):
        #Need to compute critic Values
        #Setup flat batches
        logs = {aid:{} for aid in self.to_train}
        flat_batches = {}
        flat_batches[aid] = {self.buffers[aid].get_flat_batch() for aid in self.to_train}
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size 
                mb_inds = b_inds[start:end]
                for aid in self.to_train:
                    minibatch = {k: v[mb_inds] for k, v in flat_batches[aid].items()}
                    logs[aid] = self.policies[aid].update(minibatch)
        return logs
    def compute_action(self, agent_id, obs):
        action, logprob, _, _ = self.policies[agent_id].get_action_and_value(obs)
        self.buffers[agent_id].add_logprob(logprob)
        self.buffers[agent_id].add_action(action)
        #TODO: Check return type
        return action
    
    def add_policy(self, agent_id, obs_space, act_space, policy, to_train=False, config=None):
        assert (agent_id not in self.buffers), "Currently you add two policies for one agent"
        self.obs_spaces[agent_id] = obs_space 
        self.act_spaces[agent_id] = act_space
        if not to_train:
            self.buffers[agent_id] = DummyBuffer(obs_space, act_space, self.num_envs, self.num_steps, self.device) 
        else:
            self.to_train.append(agent_id)
            
        if isinstance(policy, str):
            self.policies[agent_id] = policy
        elif isinstance(policy, Actor):
            self.policies[agent_id] = policy(obs_space, act_space, config)
            self.buffers[agent_id] = ActorBuffer(obs_space, act_space, self.num_envs, self.num_steps, self.device)
        else:
            raise NotImplementedError("Currently only supports PPO, and selfplay (policy = agent_id string)")
        
        
