#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 13:16:44 2022

@author: luke
"""

import TD3
import SAC

supported_models = ['TD3', 'SAC']

def td3_model_setup(train_env, hyperparameters, device="cpu", early_stopping:int = 100_000, verbose:int = 0):
    model = TD3.TD3(max_action=1.0, hyperparameters=hyperparameters, train_env=train_env, device=device, early_stopping=early_stopping, verbose=verbose)
    return model

def sac_model_setup(train_env, hyperparameters, device="cpu", early_stopping:int = 100_000, verbose:int = 0):
    model = SAC.SAC(max_action=1.0, hyperparameters=hyperparameters, train_env=train_env, device = device, early_stopping = early_stopping, verbose=verbose)
    return model

def setup_model( model_type: str, train_env, hyperparameters, device:str = "cpu", early_stopping:int = 100_000, verbose:int = 0 ):

    if model_type == "TD3":
        model = td3_model_setup(train_env=train_env, hyperparameters=hyperparameters, device=device, early_stopping=early_stopping, verbose=verbose) 
    elif model_type == "SAC":
        model = sac_model_setup(train_env=train_env, hyperparameters=hyperparameters, device=device, early_stopping=early_stopping, verbose=verbose) 
    else:
        print(f'{model_type} is not a supported model. Please select from {supported_models}')
        
    return model