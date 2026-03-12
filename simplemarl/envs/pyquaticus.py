
from simplemarl.policymanager import PolicyManager, MAPPO
from simplemarl.algorithms.ppo import PPOConfig, PPO, Critic, Actor



#Returns Configuration Dictionary for IPPO algorithm 
# All agents Red and Blue have individual PPO aglorithms
def IPPO_config_6():
    return {
        "policies":{
            'agent_0': {'algo':PPO, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_1': {'algo':PPO, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_2': {'algo':PPO, 'train':True, 'load_path':None, 'alg_config':PPOConfig},

            'agent_3': {'algo':PPO, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_4': {'algo':PPO, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_5': {'algo':PPO, 'train':True, 'load_path':None, 'alg_config':PPOConfig}
        }
    }

#Create IPPO Policy Handler
def IPPO(env, config):
    pm = PolicyManager()
    pa = env.possible_agents
    for aid in config['policies']:
        assert (not (aid in pa)), "Agent must be valid; i.e. exist at some point within the game"
        pm.add_policy(env.observation_space(aid),
                      env.action_space(aid),
                      config['policies'][aid]['algo'], 
                      config['policies'][aid]['train'], 
                      load=config['policies'][aid]['load_path'],
                      alg_config=config['policies'][aid]['alg_config']
                    )
        
#Returns Configuration Dictionary for MAPPO algorithm 
# All agents Red and Blue share same critic
def MAPPO_config_6():
    return {
        "policies":{
            'agent_0': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_1': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_2': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},

            'agent_3': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_4': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_5': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig}
        }
    }
#Returns Configuration Dictionary for MAPPO algorithm 
# All agents on Bule team share critic
def MAPPO_config_3_blue():
    return {
        "policies":{
            'agent_0': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_1': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_2': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
        }
    }
#Returns Configuration Dictionary for MAPPO algorithm 
# All agents on Red team share critic
def MAPPO_config_3_red():
    return {
        "policies":{
            'agent_3': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_4': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
            'agent_5': {'algo':Actor, 'train':True, 'load_path':None, 'alg_config':PPOConfig},
        }
    }

#Create MAPPO Policy Handler
def MAPPO(env, config): 
    pm = MAPPO()
    pa = env.possible_agents
    for aid in config['policies']:
        assert (not (aid in pa)), "Agent must be valid; i.e. exist at some point within the game"
        assert (config['policies'][aid]['algo'] == Actor), "MAPPO agent policies must be Actors"
        pm.add_policy(aid,
                      env.observation_space(aid),
                      env.action_space(aid),
                      config['policies'][aid]['algo'], 
                      config['policies'][aid]['train'], 
                      load=config['policies'][aid]['load_path'],
                      alg_config=config['policies'][aid]['alg_config']
                    )
        
