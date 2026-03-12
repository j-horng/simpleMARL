from simplemarl.policymanager import PolicyManager, MAPPO
from simplemarl.algorithms.ppo import PPO, Critic, Actor


def IPPO(env, agent_ids, to_train): #List of policies, list of policies to train
    pm = PolicyManager()
    for aid in agent_ids:
        if aid in to_train:
            pm.add_policy(env.observation_space(aid), env.action_space(aid), PPO, True)
        else:
            pm.add_policy(env.observation_space(aid), env.action_space(aid), PPO, False)
    
def MAPPO(env, agent_ids, to_train): #List of policies, list of policies to train
    pm = MAPPO()
    for aid in agent_ids:
        if aid in to_train:
            pm.add_policy(env.observation_space(aid), env.action_space(aid), Actor, True)
        else:
            pm.add_policy(env.observation_space(aid), env.action_space(aid), Actor, False)





def IPPO(env, agent_ids, to_train): #List of policies, list of policies to train
    pm = PolicyManager()
    for aid in agent_ids:
        if aid in to_train:
            pm.add_policy(env.observation_space(aid), env.action_space(aid), PPO, True)
        else:
            pm.add_policy(env.observation_space(aid), env.action_space(aid), PPO, False)
    
def MAPPO(env, agent_ids, to_train): #List of policies, list of policies to train
    pm = MAPPO()
    for aid in agent_ids:
        if aid in to_train:
            pm.add_policy(env.observation_space(aid), env.action_space(aid), Actor, True)
        else:
            pm.add_policy(env.observation_space(aid), env.action_space(aid), Actor, False)
