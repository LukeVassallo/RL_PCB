#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:47:01 2022

@author: luke
"""

import optuna
import json
from typing import Dict, Any
import sys

hp_opt_pi_layers_min = 1
hp_opt_pi_layers_max = 3
hp_opt_vf_layers_min = 1
hp_opt_vf_layers_max = 3
hp_opt_qf_layers_min = 1
hp_opt_qf_layers_max = 3

hp_opt_pi_neurons_min = 16
hp_opt_pi_neurons_max = 512
hp_opt_vf_neurons_min = 16
hp_opt_vf_neurons_max = 512
hp_opt_qf_neurons_min = 16
hp_opt_qf_neurons_max = 512

def set_user_attributes(study: optuna.Study):
    '''
    Sets specific parameters as user attributes to faciliate report generation
    by generalising some parameters.

    :param study: Optuna study
    :type study: optuna.Study
    :return: None
    :rtype: None

    '''
    study.set_user_attr("hp_opt_pi_layers_min", hp_opt_pi_layers_min)
    study.set_user_attr("hp_opt_pi_layers_max", hp_opt_pi_layers_max)
    study.set_user_attr("hp_opt_vf_layers_min", hp_opt_vf_layers_min)
    study.set_user_attr("hp_opt_vf_layers_max", hp_opt_vf_layers_max)
    study.set_user_attr("hp_opt_qf_layers_min", hp_opt_qf_layers_min)
    study.set_user_attr("hp_opt_qf_layers_max", hp_opt_qf_layers_max)
    study.set_user_attr("hp_opt_pi_neurons_min", hp_opt_pi_neurons_min)
    study.set_user_attr("hp_opt_pi_neurons_max", hp_opt_pi_neurons_max)
    study.set_user_attr("hp_opt_vf_neurons_min", hp_opt_vf_neurons_min)
    study.set_user_attr("hp_opt_vf_neurons_max", hp_opt_vf_neurons_max)
    study.set_user_attr("hp_opt_qf_neurons_min", hp_opt_qf_neurons_min)
    study.set_user_attr("hp_opt_qf_neurons_max", hp_opt_qf_neurons_max)

def gen_default_hyperparameters(on_policy=False):
    default_hyperparameters = {
        "learning_rate": 0.001,
        "buffer_size": 1_000_000,
        "n_steps": 2048,
        "batch_size": 128,
        "gamma": 0.99,
        "net_arch": {},
        "activation_fn": "relu",
        # Added for contsistency between all hyperparameter generator functions
        "train_freq": 1,                   # trains at the end of the episode
        "gradient_steps": 1,               # trains at the end of the episode
        # Added for multi-agent
        "tau": 0.005,                      # Target network update rate
        # Noise added to target policy during critic update
        "policy_noise": 0.2,
        "noise_clip": 0.5,                 # Range to clip target policy noise
        "policy_freq": 2,                  # Frequency of delayed policy updates
        }

    if on_policy is True:
        default_hyperparameters["net_arch"] = [dict(pi=[32, 32, 128, 64, 64],
                                                    vf=[64, 128, 64])]
        default_hyperparameters["activation_fn"] = "tanh"
    else:
        default_hyperparameters["net_arch"] = dict(pi=[400,300], qf=[400,300])

    return default_hyperparameters

def gen_default_sb3_hyperparameters(algo:str, max_steps:int):
    if algo in ("TRPO", "PPO"):
        default_hyperparameters = gen_default_hyperparameters(on_policy=True)
    else:
        default_hyperparameters = gen_default_hyperparameters(on_policy=False)

    if algo == "TRPO":
        default_hyperparameters["learning_rate"] = 0.001
        default_hyperparameters["n_steps"] = 2048
        default_hyperparameters["batch_size"] = 128

    elif algo == "PPO":
        default_hyperparameters["learning_rate"] = 0.0003
        default_hyperparameters["n_steps"] = 2048
        default_hyperparameters["batch_size"] = 64
    elif algo == "TD3":
        default_hyperparameters["learning_rate"] = 0.001
        default_hyperparameters["buffer_size"] = 1_000_000
        default_hyperparameters["batch_size"] = 100
        default_hyperparameters["train_freq"] = max_steps
        default_hyperparameters["gradient_steps"] = max_steps
        default_hyperparameters["tau"] = 0.005
        default_hyperparameters["policy_noise"] = 0.2
        default_hyperparameters["noise_clip"] = 0.5
        default_hyperparameters["policy_freq"] = 2

    elif algo == "SAC":
        default_hyperparameters["learning_rate"] = 0.0003
        default_hyperparameters["buffer_size"] = 1_000_000
        default_hyperparameters["batch_size"] = 256
        default_hyperparameters["train_freq"] = 1
        default_hyperparameters["gradient_steps"] = 1
    else:
        print(f"Algorithm {algo} is not supported. Progam terminating.")
        sys.exit()

    # common settings
    default_hyperparameters["gamma"] = 0.99

    if algo in ("TRPO", "PPO"):
        default_hyperparameters["net_arch"] = [dict(pi=[32, 32, 128, 64, 64],
                                                     vf=[64, 128, 64])]
        default_hyperparameters["activation_fn"] = "tanh"
    else:
        default_hyperparameters["net_arch"] = dict(pi=[400,300], qf=[400,300])
        default_hyperparameters["activation_fn"] = "relu"

    return default_hyperparameters

def sample_hyperparameters(trial: optuna.Trial,
                           on_policy=False) -> Dict[str, Any]:
    pi_layers = trial.suggest_int("pi_layers",
                                  hp_opt_pi_layers_min,
                                  hp_opt_pi_layers_max)
    pi_n_units_l = []
    for i in range(pi_layers):
        pi_n_units_l.append(trial.suggest_int("pi_n_units_l{}".format(i),
                                              hp_opt_pi_neurons_min,
                                              hp_opt_pi_neurons_max))

    if on_policy is True:
        vf_layers = trial.suggest_int("vf_layers",
                                      hp_opt_vf_layers_min,
                                      hp_opt_vf_layers_max)
        vf_n_units_l = []
        for i in range(vf_layers):
            vf_n_units_l.append(trial.suggest_int("vf_n_units_l{}".format(i),
                                                  hp_opt_vf_neurons_min,
                                                  hp_opt_vf_neurons_max))

        net_arch = [{"pi": pi_n_units_l, "vf": vf_n_units_l}]
    else: # off_policy
        qf_layers = trial.suggest_int("qf_layers",
                                      hp_opt_qf_layers_min,
                                      hp_opt_qf_layers_max)
        qf_n_units_l = []
        for i in range(qf_layers):
            qf_n_units_l.append(trial.suggest_int("qf_n_units_l{}".format(i),
                                                  hp_opt_qf_neurons_min,
                                                  hp_opt_qf_neurons_max))

        net_arch = {"pi": pi_n_units_l, "qf": qf_n_units_l}

    activation_fn = trial.suggest_categorical("activation_fn",
                                              ["tanh", "relu"])

    hyperparameters = gen_default_hyperparameters()
    hyperparameters["net_arch"] = net_arch
    hyperparameters["activation_fn"] = activation_fn

    return hyperparameters

def sample_hyperparameters_nas( trial: optuna.Trial,
                               algo:str,
                               max_steps:int ) -> Dict[str, Any]:
    pi_layers = trial.suggest_int("pi_layers",
                                  hp_opt_pi_layers_min,
                                  hp_opt_pi_layers_max)
    pi_n_units_l = []
    for i in range(pi_layers):
        pi_n_units_l.append(trial.suggest_int("pi_n_units_l{}".format(i),
                                              hp_opt_pi_neurons_min,
                                              hp_opt_pi_neurons_max))

    if algo in ("TRPO", "PPO"):
        vf_layers = trial.suggest_int("vf_layers",
                                      hp_opt_vf_layers_min,
                                      hp_opt_vf_layers_max)
        vf_n_units_l = []
        for i in range(vf_layers):
            vf_n_units_l.append(trial.suggest_int("vf_n_units_l{}".format(i),
                                                  hp_opt_vf_neurons_min,
                                                  hp_opt_vf_neurons_max))

        net_arch = [{"pi": pi_n_units_l, "vf": vf_n_units_l}]
    else: # off_policy
        qf_layers = trial.suggest_int("qf_layers",
                                      hp_opt_qf_layers_min,
                                      hp_opt_qf_layers_max)
        qf_n_units_l = []
        for i in range(qf_layers):
            qf_n_units_l.append(trial.suggest_int("qf_n_units_l{}".format(i),
                                                  hp_opt_qf_neurons_min,
                                                  hp_opt_qf_neurons_max))

        net_arch = {"pi": pi_n_units_l, "qf": qf_n_units_l}

    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    hyperparameters = gen_default_sb3_hyperparameters(algo=algo,
                                                      max_steps=max_steps)
    hyperparameters["net_arch"] = net_arch
    hyperparameters["activation_fn"] = activation_fn

    return hyperparameters

def sample_hyperparameters_hp( trial: optuna.Trial,
                              algo:str,
                              max_steps:int,
                              base_hyperparameters:str=None) -> Dict[str, Any]:

    if base_hyperparameters is None:
        hyperparameters = gen_default_sb3_hyperparameters(algo=algo,
                                                          max_steps=max_steps)
    else:
        hyperparameters = load_hyperparameters_from_file(base_hyperparameters)

    if algo in ("TRPO", "PPO"):
        learning_rate = trial.suggest_float("learning_rate",
                                            1e-5, 1e-2,
                                            log=True)
        batch_size = trial.suggest_categorical("batch_size",
                                               [32, 64, 128, 256, 512, 1024])
        n_steps = trial.suggest_int("n_steps",
                                    512, 4096,
                                    step=256) # 14 intervals

        hyperparameters["n_steps"] = n_steps

        pass
    else:
        learning_rate = trial.suggest_float("learning_rate",
                                            1e-5, 1e-2,
                                            log=True)
        buffer_size = trial.suggest_int("buffer_size",
                                        300_000, 3_000_000,
                                        step=300_000)
        batch_size = trial.suggest_categorical("batch_size",
                                               [32, 64, 128, 256, 512, 1024])
        # trains after every "train_freq" steps
        train_freq = trial.suggest_categorical("train_freq",
                                               [1, 2, 4, 8, 16, 32, 64, 128, 256])

        hyperparameters["learning_rate"] = learning_rate
        hyperparameters["buffer_size"] = buffer_size
        hyperparameters["train_freq"] = train_freq
        hyperparameters["gradient_steps"] = train_freq

    # common hyperparameters
    hyperparameters["learning_rate"] = learning_rate
    hyperparameters["batch_size"] = batch_size

    return hyperparameters

def load_hyperparameters_from_file(filename):
    fp = open(filename, "r", encoding="utf-8")
    hyperparameters = json.load(fp)

    return hyperparameters

def save_hyperparameters_to_file(filename, hyperparameters):
    # Write default_hyperparameters dict to a json file
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(hyperparameters, fp)

def save_best_hyperparameters(filename: str,
                              study:optuna.Study,
                              on_policy: bool = False):
    hyperparameters = gen_default_hyperparameters(on_policy=on_policy)

    best_params = study.best_params

    pi_n_units_l = []
    for i in range(best_params["pi_layers"]):
        pi_n_units_l.append(best_params[f"pi_n_units_l{i}"])

    if on_policy is True:
        vf_n_units_l = []
        for i in range(best_params["vf_layers"]):
            vf_n_units_l.append(best_params[f"vf_n_units_l{i}"])

        net_arch = [{"pi": pi_n_units_l, "vf": vf_n_units_l}]
    else:
        qf_n_units_l = []
        for i in range(best_params["qf_layers"]):
            qf_n_units_l.append(best_params[f"qf_n_units_l{i}"])

        net_arch = {"pi": pi_n_units_l, "qf": qf_n_units_l}

    hyperparameters["activation_fn"] = best_params["activation_fn"]
    hyperparameters["net_arch"] = net_arch

    save_hyperparameters_to_file(filename=filename,
                                 hyperparameters=hyperparameters)

    # Write default_hyperparameters dict to a json file
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(hyperparameters, fp)

def save_best_hyperparameters_hp(filename: str,
                                 study:optuna.Study,algo: str,
                                 max_steps: int):
    hyperparameters = gen_default_sb3_hyperparameters(algo=algo,
                                                      max_steps=max_steps)

    best_params = study.best_params
    hyperparameters["learning_rate"] = best_params["learning_rate"]
    hyperparameters["batch_size"] = best_params["batch_size"]

    if algo in ("TRPO", "PPO"):
        hyperparameters["n_steps"] = best_params["n_steps"]
    else:
        hyperparameters["buffer_size"] = best_params["buffer_size"]
        hyperparameters["train_freq"] = best_params["train_freq"]
        hyperparameters["gradient_steps"] = best_params["train_freq"]

    save_hyperparameters_to_file(filename=filename,
                                 hyperparameters=hyperparameters)

    # Write default_hyperparameters dict to a json file
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(hyperparameters, fp)

def hyperparmeters_on_policy(hyperparameters):
    '''
    Method to test whether the given hyperparameters correspond to an on_policy
    or off_policy algorithm. The neural network architecture description of an
    on_policy algorithm is stored as a python list by stable baselines3

    :param hyperparameters: hyperparameters dictionary
    :type hyperparameters: dictionary
    :return: True if hyperparameters correspond to an on-policy algorithm,\
          false otherwise
    :rtype: bool

    '''
    if type(hyperparameters["net_arch"]) == list:
        return True
    else:
        return False

def hyperparmeters_off_policy(hyperparameters):
    '''
    Method to test whether the given hyperparameters correspond to an
    off_policy or onf_policy algorithm. The neural network architecture
    description of an off_policy algorithm is stored as python dictionary
    by stable baselines3

    :param hyperparameters: hyperparameters dictionary
    :type hyperparameters: dictionary
    :return: True if hyperparameters correspond to an off-policy algorithm,\
          false otherwise
    :rtype: bool

    '''

    if type(hyperparameters["net_arch"]) == dict:
        return True
    else:
        return False
