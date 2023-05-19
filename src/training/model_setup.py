"""
The module for setting up and initializing reinforcement learning models.

This module provides functions to setup and initialize models for reinforcement
learning. Currently, it supports the TD3 (Twin Delayed DDPG) and SAC (Soft
Actor-Critic) models.

Supported models:
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic)

Module Functions:
    - td3_model_setup(train_env, hyperparameters, device="cpu",
        early_stopping: int = 100_000, verbose: int = 0):
        Setup function for the TD3 model.

    - sac_model_setup(train_env, hyperparameters, device="cpu",
        early_stopping: int = 100_000, verbose: int = 0):
        Setup function for the SAC model.

    - setup_model(model_type: str, train_env, hyperparameters,
        device: str = "cpu", early_stopping: int = 100_000, verbose: int = 0):
        Setup function to create a model based on the specified model type.
"""

import TD3
import SAC

supported_models = ["TD3", "SAC"]

def td3_model_setup(train_env,
                    hyperparameters,
                    device="cpu",
                    early_stopping:int = 100_000,
                    verbose:int = 0):
    """
    Setup function for the TD3 model.

    Args:
        train_env: The training environment.
        hyperparameters: Hyperparameters for the TD3 model.
        device (str): The device to use for computations (default: "cpu").
        early_stopping (int): The number of steps for early stopping\
              (default: 100_000).
        verbose (int): Verbosity level (0: silent, 1: intermediate output,\
              2: detailed output) (default: 0).

    Returns:
        TD3: The initialized TD3 model.
    """
    model = TD3.TD3(max_action=1.0,
                    hyperparameters=hyperparameters,
                    train_env=train_env,
                    device=device,
                    early_stopping=early_stopping,
                    verbose=verbose)
    return model

def sac_model_setup(train_env,
                    hyperparameters,
                    device="cpu",
                    early_stopping:int = 100_000,
                    verbose:int = 0):
    """
    Setup function for the SAC model.

    Args:
        train_env: The training environment.
        hyperparameters: Hyperparameters for the SAC model.
        device (str): The device to use for computations (default: "cpu").
        early_stopping (int): The number of steps for early stopping\
              (default: 100_000).
        verbose (int): Verbosity level (0: silent, 1: intermediate output,\
              2: detailed output) (default: 0).

    Returns:
        SAC: The initialized SAC model.
    """
    model = SAC.SAC(max_action=1.0,
                    hyperparameters=hyperparameters,
                    train_env=train_env,
                    device = device,
                    early_stopping = early_stopping,
                    verbose=verbose)
    return model

def setup_model( model_type: str,
                train_env,
                hyperparameters,
                device:str = "cpu",
                early_stopping:int = 100_000,
                verbose:int = 0 ):
    """
    Setup function to create a model based on the specified model type.

    Args:
        model_type (str): The type of model to setup ("TD3" or "SAC").
        train_env: The training environment.
        hyperparameters: Hyperparameters for the model.
        device (str): The device to use for computations (default: "cpu").
        early_stopping (int): The number of steps for early stopping\
            (default: 100_000).
        verbose (int): Verbosity level (0: silent, 1: intermediate output,\
            2: detailed output) (default: 0).

    Returns:
        TD3 or SAC: The initialized model based on the specified model type.

    """
    if model_type == "TD3":
        model = td3_model_setup(train_env=train_env,
                                hyperparameters=hyperparameters,
                                device=device,
                                early_stopping=early_stopping,
                                verbose=verbose)
    elif model_type == "SAC":
        model = sac_model_setup(train_env=train_env,
                                hyperparameters=hyperparameters,
                                device=device,
                                early_stopping=early_stopping,
                                verbose=verbose)
    else:
        print(f"{model_type} is not a supported model.\
              Please select from {supported_models}")

    return model
