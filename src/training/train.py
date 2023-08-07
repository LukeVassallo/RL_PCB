"""
This module is the main script for running the training of an agent using a
policy on a PCB routing environment. It imports several modules such as
environment, parameters, numpy, torch, sys, and os.

It defines several functions such as setup_seed, program_info,
log_configuration, training_run, and main. The setup_seed function sets the
seed for numpy and torch random number generators, while program_info prints
the device being used. The log_configuration function logs the training
configuration to a tensorboard file, while training_run sets up the
environment and model, trains the model, and returns the best metrics.
Finally, main reads command line arguments, performs multiple training runs,
and prints the average of the best rewards and steps for each run.

To run this script, execute the main function.
"""

from core.environment.environment import environment
from core.environment.parameters import parameters

import numpy as np
import torch
import sys
import os 
import random

from run_config import cmdline_args, write_desc_log
from hyperparameters import load_hyperparameters_from_file
from model_setup import setup_model
from callbacks import log_and_eval_callback

best_reward = -np.inf          # New best model, you could save the agent here

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)       # Seed numpy RNG
    torch.manual_seed(seed)    # Seed the RNG for all devices

    if torch.cuda.is_available():
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def program_info(device):
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"using {device}")

def log_configuration(writer, args=None, cmdline_args=None, global_step=1):
    s = str("")

    if cmdline_args is not None:
        s += str(cmdline_args)
        s += "<br><br>"

    if args is not None:
        s += str(args).replace(",","<br>")[10:-2]

    writer.add_text(tag="params", text_string=s, global_step=global_step)
    writer.flush()

def training_run(settings):
    setup_seed(seed=settings["seed"][settings["run"]])

    settings["log_dir"] = os.path.join(settings["tensorboard_dir"],
                                       settings["run_name"] + f'_{settings["run"]}_{settings["policy"]}')

    # Create directory if it doesn't exsit.
    if not os.path.isdir(settings["tensorboard_dir"]):
        os.makedirs(settings["tensorboard_dir"])

    # Create directory if it doesn't exsit.
    if not os.path.isdir(settings["log_dir"]):
        os.makedirs(settings["log_dir"])

    hp = load_hyperparameters_from_file(settings["hyperparameters"])

    env_params=parameters({"pcb_file": settings["training_pcb"],
                           "training_pcb": settings["training_pcb"],
                           "evaluation_pcb": settings["evaluation_pcb"],
                           "net": "",
                           "use_dataAugmenter": True,
                           "augment_position": True,
                           "augment_orientation": True,
                           "agent_max_action": 1,
                           "agent_expl_noise": hp["expl_noise"],
                           "debug": False,
                           "max_steps": 200,
                           "w": settings["w"],
                           "o": settings["o"],
                           "hpwl": settings["hpwl"],
                           "seed": settings["seed"][settings["run"]],
                           "ignore_power": True,
                           "log_dir": settings["log_dir"],
                           "idx": settings["pcb_idx"],
                           "shuffle_idxs": settings["shuffle_training_idxs"],
                           })

    env = environment(env_params)
    env.reset()

    model = setup_model(model_type=settings["policy"],
                        train_env=env,
                        hyperparameters=hp,
                        device=settings["device"],
                        early_stopping=settings["early_stopping"],
                        verbose=settings["verbose"])

    callback = log_and_eval_callback(log_dir=settings["log_dir"],
                                     settings=settings,
                                     hyperparameters=hp,
                                     model=model,
                                     eval_freq=settings["evaluate_every"],
                                     verbose=settings["verbose"],
                                     training_log="training.log",
                                     num_evaluations=16)

    write_desc_log( full_fn=os.path.join(settings["log_dir"],
                                         f'{settings["run_name"]}_desc.log'),
                                         settings=settings,
                                         hyperparameters=hp,
                                         model=model)

    model.explore_for_expert_targets(settings["target_exploration_steps"])
    model.learn(timesteps=settings["max_timesteps"],
                callback=callback,
                start_timesteps=settings["start_timesteps"],
                incremental_replay_buffer=settings["incremental_replay_buffer"]
                )

    return [callback.best_metrics, callback.best_mean_metrics]

def main():
    args,settings = cmdline_args()

    if settings["redirect_stdout"] is True:
        redirection_file = os.path.join(settings["tensorboard_dir"],
                                        f"{settings['policy']}_{settings['experiment']}.stdout")
        sys.stdout = open(redirection_file, "w", encoding="utf-8")

    if settings["redirect_stderr"] is True:
        redirection_file = os.path.join(settings["tensorboard_dir"],
                                        f"{settings['policy']}_{settings['experiment']}.stderr")
        sys.stderr = open(redirection_file, "w", encoding="utf-8")

    program_info(args.device)

    mean_best_rewards = []
    mean_best_steps = []
    mean_best_mean_rewards = []
    mean_best_mean_steps = []

    for run in range(settings["runs"]):
        settings["run"] = run
        perf_metrics = training_run(settings=settings)

        mean_best_rewards.append(perf_metrics[0][0])
        mean_best_steps.append(perf_metrics[0][1])
        mean_best_mean_rewards.append(perf_metrics[1][0])
        mean_best_mean_steps.append(perf_metrics[1][1])

    print(f"mean best_reward = {np.mean(mean_best_rewards)}")
    print(f"mean best_step = {np.mean(mean_best_steps)}")
    print(f"mean best_mean_reward = {np.mean(mean_best_mean_rewards)}")
    print(f"mean best_mean_step = {np.mean(mean_best_mean_steps)}")

if __name__ == "__main__":
    main()

