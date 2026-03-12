import numpy as np
import torch
from gymnasium.spaces import Box

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
        
    def add_observation(self, data:[np.ndarray, torch.Tensor]):
        if isinstance(data, np.ndarray):    
            data = torch.from_numpy(data).detach().to(self.device)

        self.observations[self.cur_step].detach().copy_(data).to(self.device)
    def add_action(self, data:[np.ndarray, torch.Tensor]):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).detach().to(self.device)
        self.actions[self.cur_step].detach().copy_(data).to(self.device)
    
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

class StateBuffer:
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
    
    #Setters
    def _add_data(self, x, attribute):
        assert isinstance(x, [np.ndarray,torch.Tensor]), "Invalid Datatype"
        if isinstance(x, np.ndarray):
            getattr(self, attribute)[self.cur_step] = torch.from_numpy(x).detach().to(self.device)
        else:
            getattr(self, attribute)[self.cur_step].detach().copy(x).to(self.device)
    
    #Added From Training Loop
    def add_state(self, x):
        self._add_data(x, "states")
    def add_value(self, x):
        self._add_data(x, "values")
    def add_reward(self, x):
        self._add_data(x, "rewards")
    def add_done(self, x):
        self._add_data(x, "dones")
    

    def get_flat_batch(self,):
        flat_states = self.observations.reshape((-1,) + self.obs_space.shape)
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

    def get_step(self,):
        return self.cur_step
    def step(self,):
        self.cur_step += 1
    def reset(self):
        #Start overwriting the old values
        self.cur_step = 0
    
    
    #Setters
    def _add_data(self, x, attribute):
        assert isinstance(x, [np.ndarray,torch.Tensor]), "Invalid Datatype"
        if isinstance(x, np.ndarray):
            getattr(self, attribute)[self.cur_step] = torch.from_numpy(x).detach().to(self.device)
        else:
            getattr(self, attribute)[self.cur_step].detach().copy(x).to(self.device)
    
    #Added From Training Loop
    def add_state(self, x):
        self._add_data(x, "states")
    def add_value(self, x):
        self._add_data(x, "values")
    def add_reward(self, x):
        self._add_data(x, "rewards")
    def add_done(self, x):
        self._add_data(x, "dones")
    

    def get_flat_batch(self,):
        flat_obs = self.observations.reshape((-1,) + self.obs_space.shape)
        flat_actions = self.advantages.reshape(-1)
        flat_logprobs = self.returns.reshape(-1)
        return {'logprobs':flat_logprobs, 'observations':flat_obs, 'actions':flat_actions}

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


class Buffer:
    def __init__(self, obs_space, act_space, num_envs, num_steps, device):
        #Variables required for setting buffer sizes
        self.num_envs = num_envs
        self.num_steps = num_steps 
        self.cur_step = 0
        self.obs_space = obs_space 
        self.act_space = act_space 
        self.device = device
        #Initialize Buffer
        self.observations = torch.zeros((self.num_steps, self.num_envs, *obs_space.shape), dtype=torch.float32)
        self.actions = torch.zeros((self.num_steps, self.num_envs, *act_space.shape), dtype=torch.float32)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        self.advantages = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        self.returns = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32)
        self.next_done = torch.zeros((self.num_envs, ), dtype=torch.float32)
        self.next_value = torch.zeros((self.num_envs,), dtype=torch.float32)
    def step(self,):
        self.cur_step += 1
    def reset(self):
        #Start overwriting the old values
        self.cur_step = 0
    #Setters
    def _add_data(self, x, attribute):
        assert isinstance(x, [np.ndarray,torch.Tensor]), "Invalid Datatype"
        if isinstance(x, np.ndarray):
            getattr(self, attribute)[self.cur_step] = torch.from_numpy(x).detach().to(self.device)
        else:
            getattr(self, attribute)[self.cur_step].detach().copy(x).to(self.device)
    
    #Added From Training Loop
    def add_observation(self, x):
        self._add_data(x, "observations")
    def add_reward(self, x):
        self._add_data(x, "rewards")
    def add_done(self, x):
        self._add_data(x, "dones")
    #Added Byproducts ()
    def add_value(self, x):
        self._add_data(x, "values")
    def add_logprob(self, x):
        self._add_data(x, "logprobs")
    
    #Getters
    def get_step(self,):
        return self.cur_step
    def get_values(self):
        return self.values[:self.cur_step].reshape(-1)
    def get_returns(self):
        return self.returns[:self.cur_step].reshape(-1)
    def get_rewards(self):
        return self.rewards[:self.cur_step]
    def get_flat_batch(self,):
        flat_obs = self.observations.reshape((-1,) + self.obs_space.shape)
        flat_logprobs = self.logprobs.reshape(-1)
        flat_actions = self.actions.reshape((-1,) + self.act_space.shape)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)
        
        return {'obs':flat_obs, 'actions':flat_actions,'logprobs':flat_logprobs,'advantages':flat_advantages,'returns':flat_returns, 'values':flat_values}

    def get_average_return(self,):
        # rew = 0.0
        # for i in range(self.dones.shape[0]):
        #     rew += self.rewards[i][0]
        #     if self.dones[i][0]:
        #         return rew
        # return rew
        return self.rewards.squeeze(-1).sum() / 40
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
    



