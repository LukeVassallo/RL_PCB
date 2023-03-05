#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 13:16:44 2022

@author: luke
"""

import TD3
import SAC

supported_models = ['TD3', 'SAC']

def td3_model_setup(train_env, hyperparameters, device="cpu", early_stopping:int = 100_000):
    # n_actions = train_env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # policy_kwargs={
    #     "net_arch": hyperparameters["net_arch"],
    #     "features_extractor_class": FeaturesExtractor,
    #     #"activation_fn": hyperparameters["activation_fn"],
    #     "activation_fn": {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[hyperparameters["activation_fn"]],
    #     }
    
    # if isSubprocVecEnv(train_env) == True:
    #     steps_to_train = int(train_env.get_attr('max_steps')[0] / train_env.num_envs)
    # else:
    #     steps_to_train = int(train_env.max_steps)
    
    # model = TD3(policy="MultiInputPolicy",
    #             policy_kwargs=policy_kwargs,
    #             env=train_env,
    #             learning_rate=hyperparameters["learning_rate"],     # default learning_rate is 0.001
    #             train_freq=(steps_to_train, 'step'),
    #             buffer_size=hyperparameters["buffer_size"],         # default buffer_size is 1000000 (size of the replay buffer)
    #             batch_size=hyperparameters["batch_size"],           # default batch_size is 100      (minibatch size for each gradient update)
    #             gamma=hyperparameters["gamma"],                     # default gamma is 0.99          (discount factor)
    #             action_noise=action_noise
    #         )
    
    # return model    
    model = TD3.TD3(max_action=1.0, hyperparameters=hyperparameters, train_env=train_env, device=device, early_stopping=early_stopping)
    return model

def sac_model_setup(train_env, hyperparameters, device="cpu", early_stopping:int = 100_000):
    # n_actions = train_env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # policy_kwargs={
    #     "net_arch": hyperparameters["net_arch"],
    #     "features_extractor_class": FeaturesExtractor,
    #     #"activation_fn": hyperparameters["activation_fn"],
    #     "activation_fn": {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[hyperparameters["activation_fn"]],
    #     }
    
    # if isSubprocVecEnv(train_env) == True:
    #     steps_to_train = int(train_env.get_attr('max_steps')[0] / train_env.num_envs)
    # else:
    #     steps_to_train = int(train_env.max_steps)
    
    # model = TD3(policy="MultiInputPolicy",
    #             policy_kwargs=policy_kwargs,
    #             env=train_env,
    #             learning_rate=hyperparameters["learning_rate"],     # default learning_rate is 0.001
    #             train_freq=(steps_to_train, 'step'),
    #             buffer_size=hyperparameters["buffer_size"],         # default buffer_size is 1000000 (size of the replay buffer)
    #             batch_size=hyperparameters["batch_size"],           # default batch_size is 100      (minibatch size for each gradient update)
    #             gamma=hyperparameters["gamma"],                     # default gamma is 0.99          (discount factor)
    #             action_noise=action_noise
    #         )
    
    # return model    
    #model = TD3.TD3(state_dim=23, action_dim=3, max_action=1.0, hyperparameters=hyperparameters)
    model = SAC.SAC(max_action=1.0, hyperparameters=hyperparameters, train_env=train_env, device = device, early_stopping = early_stopping)
    return model

def setup_model( model_type: str, train_env, hyperparameters, device:str = "cpu", early_stopping:int = 100_000 ):

    if model_type == "TD3":
        model = td3_model_setup(train_env=train_env, hyperparameters=hyperparameters, device=device, early_stopping=early_stopping) 
    elif model_type == "SAC":
        model = sac_model_setup(train_env=train_env, hyperparameters=hyperparameters, device=device, early_stopping=early_stopping) 
    else:
        print(f'{model_type} is not a supported model. Please select from {supported_models}')
        
    return model