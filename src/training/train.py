from core.environment.environment import environment
from core.environment.parameters import parameters

import numpy as np
import torch
import os
import datetime
import random

from run_config import cmdLine_args, write_desc_log
from hyperparameters import load_hyperparameters_from_file
from model_setup import setup_model
from callbacks import log_and_eval_callback

best_reward = -np.inf          # New best model, you could save the agent here

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)       # Seed numpy RNG  
    torch.manual_seed(seed)    # seed the RNG for all devices (both CPU and CUDA)

    if torch.cuda.is_available():
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def program_info(device):    
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    print(f'using {device}')

def log_configuration(writer, args=None, cmdline_args=None, global_step=1):
    s = str("")

    if cmdline_args is not None:
        s += str(cmdline_args)
        s += "<br><br>"

    if args is not None:
        s += str(args).replace(',',"<br>")[10:-2]
                
    writer.add_text(tag="params", text_string=s, global_step=global_step)
    writer.flush()        

def evaluate_save_video(policy, seed, video_path, args, pcb_file, model_path=None, eval_episodes=10, writer=None, run_name=0, t=0, target_params=None, write_pcb_file=False, output_pcb_file_path=None, save_best_layouts=False):
    global best_reward
    
    file_best_hpwl_zero_overlap = "best_hpwl_zero_overlap"
    file_best_hpwl_10_overlap = "best_hpwl_10_overlap"
    file_best_hpwl_20_overlap = "best_hpwl_20_overlap"
    
    best_hpwl = 1E6
    best_hpwl_at_10_overlap = 1E6
    best_hpwl_at_20_overlap = 1E6

    params_dict = {         
        "pcb_file": pcb_file,
        "net": "/home/luke/Desktop/semi_autonomous/py/pcb_component_w_vec_distance_v2/reward_v5/05_log/1656941074/1656941074_best_model.td3",
        "use_dataAugmenter": True,
        "augment_position": True,
        "augment_orientation": True,
        "agent_max_action": 1,
        "agent_expl_noise": args.expl_noise,
        "debug": True,
        "w": args.w,
        "o": args.o,
        "hpwl": args.hpwl,
        "seed": 3142,
        "ignore_power": True,
        "log_dir": video_path,
        "shuffle_idxs": args.shuffle_evaluation_idxs
        }

    env_params=parameters(params_dict)
    
    eval_env = environment(env_params)
    
    snapshot_dir=os.path.join(video_path, "snapshots")
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)  

    total_reward=0
    
    evaluation_log = open(os.path.join(video_path,"evaluation.log"), "w")
    evaluation_log.write(f'timestamp={datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}\r\n')
    evaluation_log.write('parameters begin\r\n')
    for key, value in params_dict.items():
        evaluation_log.write(f'\t{key}={value}\r\n')
    evaluation_log.write('parameters end\r\n')        

    for i in range(eval_episodes):
        
        best_hpwl = 1E6
        best_hpwl_at_10_overlap = 1E6
        best_hpwl_at_20_overlap = 1E6
        
        eval_env.reset()
        done = False
        episode_steps=0
        while not done:
            episode_steps += 1
            obs_vec = eval_env.step(model=policy, random=False)
            step_reward=0
            
            if save_best_layouts == True:
                # measure wirelength and overlap.
                all_ol = []
                hpwl = eval_env.calc_hpwl()
                
                for indiv_obs in obs_vec:
                    all_ol.append(np.max(indiv_obs[1][8:15]))
                    
                if (hpwl < best_hpwl) and (np.max(all_ol) < 1E-6):
                    best_hpwl = hpwl
                    filename = file_best_hpwl_zero_overlap + f'_{int(t)}k_{i}.{eval_episodes-1}.{episode_steps}.pcb' 
                    eval_env.write_current_pcb_file(path=output_pcb_file_path, filename=filename)
                    evaluation_log.write(f'run={i}/{eval_episodes-1} @ episode_step={episode_steps} : Zero overlap best hpwl : hpwl={np.round(hpwl,4)}, overlap={np.round(np.sum(all_ol)/8,4)}\r\n')
                    evaluation_log.write(f'all_ol={all_ol}\r\n')

                    # Capture snapshot
                    snapshot_filename=f'{i}.{eval_episodes-1}.{episode_steps}'
                    eval_env.tracker.capture_snapshot(fileName=os.path.join(snapshot_dir, snapshot_filename+".png"))
                    eval_env.write_current_pcb_file(path=snapshot_dir, filename= snapshot_filename+".pcb")  # Yes this is exactly like the previous one.
                                        
                    print(f'run={i}/{eval_episodes-1} @ episode_step={episode_steps} : Zero overlap best hpwl : hpwl={np.round(hpwl,4)}, overlap={np.round(np.sum(all_ol)/8,4)}')
                    
                if (hpwl < best_hpwl_at_10_overlap ) and (np.max(all_ol) <= 0.1):
                    best_hpwl_at_10_overlap = hpwl
                    filename = file_best_hpwl_10_overlap + f'_{int(t)}k_{i}.{eval_episodes-1}.{episode_steps}.pcb' 
                    eval_env.write_current_pcb_file(path=output_pcb_file_path, filename=filename)
                    evaluation_log.write(f'run={i}/{eval_episodes-1} @ episode_step={episode_steps} : 10% overlap best hpwl : hpwl={np.round(best_hpwl_at_10_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}\r\n')
                    evaluation_log.write(f'all_ol={all_ol}\r\n')
                    
                    # Capture snapshot
                    snapshot_filename=f'{i}.{eval_episodes-1}.{episode_steps}'
                    eval_env.tracker.capture_snapshot(fileName=os.path.join(snapshot_dir, snapshot_filename+".png"))
                    eval_env.write_current_pcb_file(path=snapshot_dir, filename= snapshot_filename+".pcb")  # Yes this is exactly like the previous one.
                    
                    print(f'run={i}/{eval_episodes-1} @ episode_step={episode_steps} : 10% overlap best hpwl : hpwl={np.round(best_hpwl_at_10_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}')

                    
                if (hpwl < best_hpwl_at_20_overlap ) and (np.max(all_ol) <= 0.2):
                    best_hpwl_at_20_overlap = hpwl
                    filename = file_best_hpwl_20_overlap + f'_{int(t)}k_{i}.{eval_episodes-1}.{episode_steps}.pcb' 
                    eval_env.write_current_pcb_file(path=output_pcb_file_path, filename=filename)
                    evaluation_log.write(f'run={i}/{eval_episodes-1} @ episode_step={episode_steps} : 20% overlap best hpwl : hpwl={np.round(best_hpwl_at_20_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}\r\n')
                    evaluation_log.write(f'all_ol={all_ol}\r\n')
                    
                    # Capture snapshot
                    snapshot_filename=f'{i}.{eval_episodes-1}.{episode_steps}'
                    eval_env.tracker.capture_snapshot(fileName=os.path.join(snapshot_dir, snapshot_filename+".png"))
                    eval_env.write_current_pcb_file(path=snapshot_dir, filename= snapshot_filename+".pcb")  # Yes this is exactly like the previous one.

                    print(f'run={i}/{eval_episodes-1} @ episode_step={episode_steps} : 20% overlap best hpwl : hpwl={np.round(best_hpwl_at_20_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}')

            
            for indiv_obs in obs_vec:
                step_reward += indiv_obs[2]
                if indiv_obs[4] == True:
                    done = True
                
            step_reward /= len(obs_vec)
            total_reward += step_reward
        evaluation_log.write(f'eval_env episode {i} performed {episode_steps} in environment.\r\n')
        print(f'eval_env episode {i} performed {episode_steps} in environment.')
        eval_env.tracker.create_video(fileName=os.path.join(video_path, f'{i}.mp4'))
        eval_env.tracker.create_plot(fileName=os.path.join(video_path, f'{i}.png'))
        eval_env.tracker.log_run_to_file(path=video_path, filename=f'{i}.log', kicad_pcb=eval_env.g.get_kicad_pcb_file())
        eval_env.tracker.reset()
    
    evaluation_log.close()
    
    if model_path is not None:     
        policy.save(filename=os.path.join(model_path, f'policy_{int(t/1000)}k.td3'))
            
        if (total_reward / eval_episodes) > best_reward:
            best_reward = total_reward / eval_episodes
            print(f"New best with average reward of {best_reward}")
            policy.save(filename=os.path.join(model_path, 'policy_best.td3'))
            
    if write_pcb_file and output_pcb_file_path is not None:
        
        for i in range(len(eval_env.pv)):
            g = eval_env.pv[i].get_graph()
            g.update_original_nodes_with_current_optimals()
        eval_env.write_pcb_file(path=output_pcb_file_path, filename=f'{int(t)}k.pcb')
    return

def training_run(settings):   
    setup_seed(seed=settings['seed'][settings["run"]])
    
    settings['log_dir'] = os.path.join(settings['tensorboard_dir'], settings['run_name'] + f'_{settings["run"]}_{settings["policy"]}')

    if os.path.isdir(settings["tensorboard_dir"]) == False: # Create directory if it doesn't exsit.
        os.makedirs(settings["tensorboard_dir"])
        
    if os.path.isdir(settings["log_dir"]) == False: # Create directory if it doesn't exsit.
        os.makedirs(settings["log_dir"]) 
        
    hp = load_hyperparameters_from_file(settings["hyperparameters"])

    env_params=parameters({"pcb_file": settings["training_pcb"],
                           "training_pcb": settings["training_pcb"],
                           "evaluation_pcb": settings["evaluation_pcb"],
                           "net": "/home/luke/Desktop/semi_autonomous/py/pcb_component_w_vec_distance_v2/reward_v5/05_log/1656941074/1656941074_best_model.td3",
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
                           "seed": settings['seed'][settings["run"]],
                           "ignore_power": True,
                           "log_dir": settings['log_dir'],
                           "idx": None,
                           "shuffle_idxs": settings['shuffle_training_idxs'],
                           })
    
    env = environment(env_params)
    env.reset()
    
    model = setup_model(model_type=settings["policy"], train_env=env, hyperparameters=hp, device=settings["device"], early_stopping=settings["early_stopping"], verbose=settings["verbose"])          

    callback = log_and_eval_callback(log_dir=settings['log_dir'],
                                     settings=settings,
                                     hyperparameters=hp,
                                     model=model,
                                     eval_freq=settings['evaluate_every'],
                                     verbose=settings['verbose'],
                                     training_log="training.log",
                                     num_evaluations=16,  
                                     )
    
    write_desc_log( full_fn=os.path.join(settings["log_dir"], f'{settings["run_name"]}_desc.log'), settings=settings, hyperparameters=hp, model=model)
    
    model.explore_for_expert_targets(settings["target_exploration_steps"])
    model.learn(timesteps=settings["max_timesteps"], callback=callback, start_timesteps=settings["start_timesteps"], incremental_replay_buffer=settings["incremental_replay_buffer"])
    
    return [callback.best_metrics, callback.best_mean_metrics]

def main():
    args,settings = cmdLine_args()
    
    program_info(args.device)
        
    mean_best_rewards = []
    mean_best_steps = []
    mean_best_mean_rewards = []
    mean_best_mean_steps = []
    

    for run in range(settings["runs"]):
        settings['run'] = run
        perf_metrics = training_run(settings=settings)
        
        mean_best_rewards.append(perf_metrics[0][0])
        mean_best_steps.append(perf_metrics[0][1])
        mean_best_mean_rewards.append(perf_metrics[1][0])
        mean_best_mean_steps.append(perf_metrics[1][1])
                
    print(f'mean best_reward = {np.mean(mean_best_rewards)}')
    print(f'mean best_step = {np.mean(mean_best_steps)}')
    print(f'mean best_mean_reward = {np.mean(mean_best_mean_rewards)}')
    print(f'mean best_mean_step = {np.mean(mean_best_mean_steps)}')
        
if __name__ == "__main__":
    main()
