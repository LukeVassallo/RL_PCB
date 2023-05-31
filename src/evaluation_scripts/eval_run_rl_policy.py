"""
This module provides functions for evaluating a multi-agent PCB component
placement using reinforcement learning.

It includes the following functions:

- configure_seed(args): Configure the seed values for random number generation
based on the provided arguments.
- cmdline_args(): Parse command line arguments and return the settings for
evaluation.
- set_seed_everywhere(seed): Set the seed for random number generation in
various libraries.
- evaluation_run(settings): Perform the evaluation run using the given settings.

Note: This module requires the following dependencies: os, sys, pathlib,
argparse, datetime, torch, numpy, and random.

Usage example:
args, settings = cmdline_args()
evaluation_run(settings)
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import torch
import numpy as np
import random

sys.path.append("../training")
from core.environment.environment import environment
from core.environment.utils import get_pcb_num
from core.environment.parameters import parameters
from model_setup import setup_model
from hyperparameters import load_hyperparameters_from_file

def configure_seed(args):
    """
    Configure the seed values for random number generation based on the
    provided arguments.

    Args:
        args: An object containing the configuration arguments.

    Returns:
        None

    Raises:
        None

    Notes:
        - If `args.auto_seed` is True and a valid seed configuration is
        provided, the auto_seed takes precedence and will override the
        provided seed values.
        - If `args.auto_seed` is True, the run seed values will be assigned
        randomly based on an RNG seed generated with the current time.
        - If `args.auto_seed` is False, a seed value should be provided for
        each run, or a warning will be issued.

    Example:
        configure_seed(args)
    """
    if (args.auto_seed is True) and (args.seed is not None):
        if len(args.seed) == args.runs:
            print("auto_seed is enabled while a valid seed configuration was provided. auto_seed takes precedence and will override the provided seed values.")

    # assign run seed values randomly based of an rng seed with current time.
    if args.auto_seed is True:
        args.seed = []
        rng = np.random.default_rng(seed=int(datetime.now().strftime("%s")))
        for _ in range(args.runs):
            args.seed.append(
                np.int0(rng.uniform(low=0,high=(np.power(2,32)-1)))
                )
    else:
        # seed value is not provided or not provided correctly
        if (args.seed is None) or (len(args.seed) != args.runs):
            # issue a warning
            rng = np.random.default_rng(seed=99)
            args.seed = []
            for _ in range(args.runs):
                args.seed.append(
                    np.int0(rng.uniform(low=0,high=(np.power(2,32)-1)))
                    )

def cmdline_args():
    parser = argparse.ArgumentParser(
        description="Multi-agent pcb component placement evaluation",
        usage="<script-name> -p <pcb_file> --rl_model_type [TD3 | SAC]",
        epilog="This text will be shown after the help")
    parser.add_argument("--policy", type=str, choices=["TD3", "SAC"],
                        required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pcb_file", type=str, required=True)
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("-s", "--seed", required=False, nargs="+",
                        type=np.uint32, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        required=False)
    parser.add_argument("--hyperparameters", required=False, type=str, default="auto", help="path to a hyperparameters file. If auto, the hyperparamters file will be retrived from the description file.")
    # Max time steps to run environment (1e6)
    parser.add_argument("--max_steps", type=int, default=int(1e3),
                        required=False)
    parser.add_argument("--runs", required=False, type=int, default=1,
                        help="Number of times to run the experiment.")
    parser.add_argument("--auto_seed", required=False, action="store_true", default=False, help="ignore seed value and generate one based of the current time for everyrun")
    parser.add_argument("--verbose", required=False, type=int, default=0,
                        help="Program verbosity")
    parser.add_argument("-o", "--output", required=False, type=str,
                        default="./eval_output")
    parser.add_argument("--quick_eval", required=False, action="store_true",
                        default=False,
                        help="If true video and run log won't be generated.")
    parser.add_argument("--reward_params",  type=str, required=True,
                        help="Colon seperated weights for euclidean wirelength, hpwl and overlap")
    parser.add_argument("--shuffle_idxs", required=False, action="store_true",
                        help="shuffle agent idx prior to stepping in the environment")

    args = parser.parse_args()
    settings = {}
    settings["default_seed"] = args.seed

    configure_seed(args)

    settings["policy"] = args.policy
    settings["model"] = args.model
    settings["pcb_file"] = args.pcb_file
    settings["seed"] = args.seed
    settings["max_steps"] = args.max_steps

    if args.device == "cuda":
        settings["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        print("Warning the command line argument specified a CUDA device,\
               but none were found. Using CPU.")
        settings["device"] = "cpu"

    settings["hyperparameters"] = args.hyperparameters
    settings["runs"] = args.runs
    settings["run_name"] = datetime.now().strftime("%s")
    settings["verbose"] = args.verbose
    settings["output"] = args.output
    settings["quick_eval"] = args.quick_eval

    rp = args.reward_params.split(":")

    settings["w"] = float(rp[0])       # euclidean wirelength
    settings["hpwl"] = float(rp[1])    # hpwl
    settings["o"] = float(rp[2])       # overlap

    settings["shuffle_idxs"] = args.shuffle_idxs
    return args, settings

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def evaluation_run(settings):
    hp = load_hyperparameters_from_file(settings["hyperparameters"])

    model = setup_model(model_type=settings["policy"],
                        train_env=None,
                        hyperparameters=hp,
                        device=settings["device"])
    try:
        model.load(settings["model"])
    except:
        print("Failed to load model. Does the hyperparameters file correspond\
               to the model file?")
        sys.exit()

    Path(settings["output"]).mkdir(parents=True, exist_ok=True)

    for j in range(get_pcb_num(settings["pcb_file"])):
        total_reward=0
        total_steps = 0

        file_best_hpwl_00_overlap = "best_hpwl_00_overlap"
        file_best_hpwl_10_overlap = "best_hpwl_10_overlap"
        file_best_hpwl_20_overlap = "best_hpwl_20_overlap"

        best_hpwl = 1E6
        best_hpwl_at_10_overlap = 1E6
        best_hpwl_at_20_overlap = 1E6

        set_seed_everywhere(settings["seed"][settings["run"]])

        env_params=parameters({
            "pcb_file": settings["pcb_file"],
            "training_pcb": None,
            "evaluation_pcb": None,
            "net": "",
            "use_dataAugmenter": True,
            "augment_position": True,
            "augment_orientation": True,
            "agent_max_action": 1,
            "agent_expl_noise": hp["expl_noise"],
            "debug": True,
            "max_steps": settings["max_steps"],
            "w": settings["w"],
            "o": settings["o"],
            "hpwl": settings["hpwl"],
            "seed": settings["seed"][settings["run"]],
            "ignore_power": True,
            "log_dir": None,
            "idx": j,
            "shuffle_idxs": settings["shuffle_idxs"],
            })

        eval_env = environment(env_params)

        # create directories
        current_pcb_filename = eval_env.get_current_pcb_name()
        pcb_dir = os.path.join(settings["output"], current_pcb_filename)
        Path(pcb_dir).mkdir(parents=True, exist_ok=True)

        sa_dir = os.path.join(pcb_dir, "sa_pcb")
        Path(sa_dir).mkdir(parents=True, exist_ok=True)

        run_output_dir = os.path.join(pcb_dir,f"trial_{settings['run']}")
        Path(run_output_dir).mkdir(parents=True, exist_ok=True)

        # start the evaluation log
        evaluation_log = open(os.path.join(run_output_dir,"evaluation.log"),
                              "w",
                              encoding="utf-8")
        evaluation_log.write(
            f"timestamp={datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')}\r\n")
        evaluation_log.write("parameters begin\r\n")
        evaluation_log.write(eval_env.parameters.to_text_string(prefix="\t"))
        evaluation_log.write("parameters end\r\n")

        eval_env.reset()
        eval_env.write_current_pcb_file(path=sa_dir,
                                        filename=current_pcb_filename+f"_{settings['run']}.pcb")
        done = False
        episode_steps=0
        while not done:
            episode_steps += 1
            if settings["policy"] == "TD3":
                obs_vec = eval_env.step(model=model.actor,
                                        random=False,
                                        deterministic=True,
                                        rl_model_type="TD3")
                step_reward=0
            else:   # SAC
                obs_vec = eval_env.step(model=model.policy,
                                        random=False,
                                        deterministic=True,
                                        rl_model_type="SAC")
                step_reward=0

            for indiv_obs in obs_vec:
                step_reward += indiv_obs[2]
                if indiv_obs[4] is True:
                    done = True

            step_reward /= len(obs_vec)
            total_reward += step_reward
            total_steps += 1

            hpwl = eval_env.calc_hpwl()

            all_ol = []
            for indiv_obs in obs_vec:
                next_state_ol = indiv_obs[1][8:16]
                ol_ratios = indiv_obs[-1]["ol_ratios"]
                all_ol.append(
                    np.sum(np.array(next_state_ol)*np.array(ol_ratios))*100)

            if (hpwl < best_hpwl) and (np.max(all_ol) < 1):
                best_hpwl = hpwl
                filename = file_best_hpwl_00_overlap + f"{settings['run']}.{settings['runs']-1}.{episode_steps}.pcb"
                eval_env.write_current_pcb_file(path=run_output_dir, filename=filename)
                evaluation_log.write(f"run={settings['run']}/{settings['runs']-1} @ episode_step={episode_steps} : Zero overlap best hpwl : hpwl={np.round(hpwl,4)}, overlap={np.round(np.sum(all_ol)/8,4)}\r\n")
                evaluation_log.write(f"all_ol={all_ol}\r\n")

                # Capture snapshot
                snapshot_filename=f'{settings["run"]}.{settings["runs"]-1}.{episode_steps}'
                eval_env.tracker.capture_snapshot(fileName=os.path.join(run_output_dir, snapshot_filename+".png"))
                # Yes this is exactly like the previous one.
                eval_env.write_current_pcb_file(
                    path=run_output_dir,
                    filename= snapshot_filename+".pcb")
                # overwrite best; unique filename for easier processing with
                # automated tools
                eval_env.write_current_pcb_file(
                    path=run_output_dir,
                    filename=file_best_hpwl_00_overlap+".pcb")

                if settings["verbose"] == 1:
                    print(f"run={settings['run']}/{settings['runs']-1} @ episode_step={episode_steps} : Zero overlap best hpwl : hpwl={np.round(hpwl,4)}, overlap={np.round(np.sum(all_ol)/8,4)}")

            if (hpwl < best_hpwl_at_10_overlap ) and (np.max(all_ol) <= 10.0):
                best_hpwl_at_10_overlap = hpwl
                filename = file_best_hpwl_10_overlap + f"{settings['run']}.{settings['runs']-1}.{episode_steps}.pcb"
                eval_env.write_current_pcb_file(path=run_output_dir,
                                                filename=filename)
                evaluation_log.write(f"run={settings['run']}/{settings['runs']-1} @ episode_step={episode_steps} : 10% overlap best hpwl : hpwl={np.round(best_hpwl_at_10_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}\r\n")
                evaluation_log.write(f"all_ol={all_ol}\r\n")

                # Capture snapshot
                snapshot_filename=f'{settings["run"]}.{settings["runs"]-1}.{episode_steps}'
                eval_env.tracker.capture_snapshot(fileName=os.path.join(run_output_dir, snapshot_filename+".png"))
                # Yes this is exactly like the previous one.
                eval_env.write_current_pcb_file(
                    path=run_output_dir,
                    filename= snapshot_filename+".pcb")
                # overwrite best; unique filename for easier processing with
                # automated tools
                eval_env.write_current_pcb_file(
                    path=run_output_dir,
                    filename=file_best_hpwl_10_overlap+".pcb")

                if settings["verbose"] == 1:
                    print(f'run={settings["run"]}/{settings["runs"]-1} @ episode_step={episode_steps} : 10% overlap best hpwl : hpwl={np.round(best_hpwl_at_10_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}')

            if (hpwl < best_hpwl_at_20_overlap ) and (np.max(all_ol) <= 20.0):
                best_hpwl_at_20_overlap = hpwl
                filename = file_best_hpwl_20_overlap + f'{settings["run"]}.{settings["runs"]-1}.{episode_steps}.pcb'
                eval_env.write_current_pcb_file(path=run_output_dir,
                                                filename=filename)
                evaluation_log.write(f"run={settings['run']}/{settings['runs']-1} @ episode_step={episode_steps} : 20% overlap best hpwl : hpwl={np.round(best_hpwl_at_20_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}\r\n")
                evaluation_log.write(f"all_ol={all_ol}\r\n")

                # Capture snapshot
                snapshot_filename=f"{settings['run']}.{settings['runs']-1}.{episode_steps}"
                eval_env.tracker.capture_snapshot(fileName=os.path.join(run_output_dir, snapshot_filename+".png"))
                # Yes this is exactly like the previous one.
                eval_env.write_current_pcb_file(
                    path=run_output_dir,
                    filename=snapshot_filename+".pcb")
                # overwrite best; unique filename for easier processing with
                # automated tools
                eval_env.write_current_pcb_file(
                    path=run_output_dir,
                    filename=file_best_hpwl_20_overlap+".pcb")

                if settings["verbose"] == 1:
                    print(f"run={settings['run']}/{settings['runs']-1} @ episode_step={episode_steps} : 20% overlap best hpwl : hpwl={np.round(best_hpwl_at_20_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}")

        if settings["verbose"] == 1:
            print(f"eval_env episode {settings['run']} performed {episode_steps} in environment.")

        evaluation_log.write(f"eval_env episode {settings['run']} performed {episode_steps} steps in environment.\r\n")
        evaluation_log.close()

        if settings["quick_eval"] is False:
            eval_env.tracker.create_video(
                fileName=os.path.join(run_output_dir,
                                      f"{settings['run']}.mp4")
                                      )
            eval_env.tracker.log_run_to_file(
                path=run_output_dir,
                filename=f"{settings['run']}.log",
                kicad_pcb=eval_env.g.get_kicad_pcb_file())

        eval_env.tracker.reset()

def main():
    _ , settings = cmdline_args()

    for run in range(settings["runs"]):
        settings["run"] = run
        evaluation_run(settings)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0) # success
    except:
        sys.exit(-1) # fail
