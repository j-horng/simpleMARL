import os
import time 
from dataclasses import dataclass, field
import tyro 
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
#Buffer/ML algorithms
import numpy as np 
import torch 
import random 
from simplemarl.vecenv import SerialVecEnv, ParallelVecEnv, SubProcVecEnv
from simplemarl.algorithms import ppo
from simplemarl.buffer import Buffer
from simplemarl.parallel_pet_wrapper import GymnasiumToPettingZooParallel
#Environment Imports
# from maritime_env import MaritimeRaceEnv
from pyquaticus import pyquaticus_v0
from pyquaticus.mctf26_config import config_dict_std as mctf_config

from pyquaticus.envs.competition_pyquaticus import CompPyquaticusEnv

import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import argparse

def make_env():
    import pyquaticus.utils.rewards as rew
    rews = {'agent_0':rew.caps_and_grabs,
            'agent_1':rew.caps_and_grabs,
            'agent_2':rew.caps_and_grabs,
            'agent_3':rew.caps_and_grabs,
            'agent_4':rew.caps_and_grabs,
            'agent_5':rew.caps_and_grabs}
    mc_config = dict(mctf_config)
    mc_config['render_saving'] = True
    mc_config['max_time'] = 50000.0
    env = CompPyquaticusEnv(render_mode='human', config_dict=mctf_config, reward_config=rews)
    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 3v3 PyQuaticus environment')
    parser.add_argument('agent_0', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-0-policy')
    parser.add_argument('agent_1', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-1-policy') 
    parser.add_argument('agent_2', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-1-policy') 
    parser.add_argument('--defender', default="", help='Optional path to a defender checkpoint. If set, agent_3/agent_4/agent_5 will share this fixed defender policy. If omitted, defenders mirror attackers (agent_3=agent_0, agent_4=agent_1, agent_5=agent_2).')
    parser.add_argument('--static_defenders', action='store_true', help='If set, defenders (agent_3/agent_4/agent_5) will take a constant zero/no-op action every step (they will not move). Overrides --defender.')
    parser.add_argument('--one_defender', action='store_true', help='If set, only agent_3 will be an active defender; agent_4 and agent_5 will be static. Useful for testing attacker flag captures.')
    parser.add_argument('--stack_defenders', action='store_true', help='If set, all defenders (agent_3/agent_4/agent_5) will spawn at the same fixed location at reset. Best used with --static_defenders to keep them immobile.')
    parser.add_argument('--defender_spawn_xy', type=float, nargs=2, default=None, metavar=('X', 'Y'), help='Override defender spawn location used by --stack_defenders. Coordinates are in env units (meters).')
    args = parser.parse_args()
    env = make_env()
    policies = {
        'agent_0': PPO(env.observation_space('agent_0'), env.action_space('agent_0')),
        'agent_1': PPO(env.observation_space('agent_1'), env.action_space('agent_1')),
        'agent_2': PPO(env.observation_space('agent_2'), env.action_space('agent_2')),
    }
    defender_policy = PPO(env.observation_space('agent_3'), env.action_space('agent_3'))
    policies['agent_0'].load_state_dict(torch.load(args.agent_0))
    policies['agent_1'].load_state_dict(torch.load(args.agent_1))
    policies['agent_2'].load_state_dict(torch.load(args.agent_2))
    if args.defender:
        defender_policy.load_state_dict(torch.load(args.defender))

    reset_options = None
    if args.stack_defenders:
        if args.defender_spawn_xy is not None:
            # Use plain Python types (not numpy arrays). Pyquaticus init_dict parsing
            # compares to None with `==`, which breaks on numpy arrays.
            spawn_xy = [float(args.defender_spawn_xy[0]), float(args.defender_spawn_xy[1])]
        else:
            # Default: near the red side, centered vertically.
            spawn_xy = [float(env.env_size[0] - 10.0), float(env.env_size[1] / 2.0)]
        init_dict = {
            "agent_position": {
                "agent_3": list(spawn_xy),
                "agent_4": list(spawn_xy),
                "agent_5": list(spawn_xy),
            },
            "agent_pos_unit": "m",
        }
        reset_options = {"init_dict": init_dict}

    obs, _ = env.reset(options=reset_options)
    terms = {'agent_0':False}
    rsum = {'agent_0':0.0, 'agent_1':0.0, 'agent_2':0.0, 'agent_3':0.0, 'agent_4':0.0, 'agent_5':0.0}
    steps = 0
    # print("Agent_0 Obs: ", obs['agent_0'] == obs['agent_5'])
    # print("Agent_1 Obs: ", obs['agent_1'] == obs['agent_4'])
    # print("Agent_2 Obs: ", obs['agent_2'] == obs['agent_3'])
    # time.sleep(5)
    # sys.exit()
    while not any(terms.values()):
        actions = {}
        for aid in obs:
            with torch.no_grad():
                if aid == "agent_0":
                    actions[aid] = policies["agent_0"].get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy()
                elif aid == "agent_1":
                    actions[aid] = policies["agent_1"].get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy()
                elif aid == "agent_2":
                    actions[aid] = policies["agent_2"].get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy()
                elif aid in ("agent_3", "agent_4", "agent_5"):
                    # If we only want one active defender, keep agent_4/5 static.
                    if args.one_defender and aid in ("agent_4", "agent_5"):
                        act_space = env.action_space(aid)
                        if hasattr(act_space, "n"):
                            actions[aid] = 0
                        else:
                            actions[aid] = np.zeros(act_space.shape, dtype=np.float32)
                        continue

                    if args.static_defenders:
                        act_space = env.action_space(aid)
                        if hasattr(act_space, "n"):
                            actions[aid] = 0
                        else:
                            actions[aid] = np.zeros(act_space.shape, dtype=np.float32)
                    elif args.defender:
                        actions[aid] = defender_policy.get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy()
                    elif aid == "agent_3":
                        actions[aid] = policies["agent_0"].get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy()
                    elif aid == "agent_4":
                        actions[aid] = policies["agent_1"].get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy()
                    else:
                        actions[aid] = policies["agent_2"].get_action_and_value(torch.from_numpy(obs[aid]))[0].detach().cpu().numpy()
        obs, rews, terms, truncs, _ = env.step(actions)
        print(f"Rewards: {rews}")
        for aid in rsum:
            rsum[aid] += rews[aid]
        steps += 1
    print(f"Finale sum: {rsum} Steps: {steps}")
    #Save Video
    env.buffer_to_video(recording_compression=True)
