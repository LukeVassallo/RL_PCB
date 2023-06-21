import os
import copy
import numpy as np
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from core.environment.environment import environment

from pcb import pcb
from graph import graph     # Necessary for graph related methods

class log_and_eval_callback():
    def __init__(
            self,
            log_dir: str,
            settings: dict,
            hyperparameters: dict,
            model,
            verbose: int = 1,
            num_evaluations: int = 10,
            eval_freq: int = 10_000,
            training_log: str = None,
        ):

        super().__init__()
        self.log_dir = log_dir
        # root logging directory
        self.save_path = self.log_dir
        # subdirectory containing saved models
        self.model_path = os.path.join(self.save_path, "models")
        # subdirectory containing evaluations
        self.video_path = os.path.join(self.save_path, "video_dir")
        # subdirectory containing evaluations based of training pcb file
        self.video_train_path = os.path.join(self.video_path, "training_set")
        # subdirectory containing evaluations based off evaluation pcb file
        self.video_eval_path = os.path.join(self.video_path, "evaluation_set")
        # optimals file for saving best We.
        self.optimals = os.path.join(self.save_path, "wirelength.optimals")

        self.verbose = verbose
        self.num_evaluations = num_evaluations
        self.eval_freq = eval_freq
        # logfile to save episode progress
        if training_log is not None:
            self.training_log = os.path.join(self.save_path,training_log)
        else:
            self.training_log = training_log    # None

        # Store the number of lifetime episodes.
        self.num_episodes = 0
        self.last_rollout_finish = None

        self.reward = 0

        self.best_episode_reward = -np.inf
        self.best_mean_episode_reward = -np.inf
        self.last_best_mean_timestep = 0

        # Create directory if it doesn't exsit.
        if os.path.isdir(self.log_dir) is False:
            os.makedirs(self.log_dir)

        # Create directory if it doesn't exsit.
        if os.path.isdir(self.save_path) is False:
            os.makedirs(self.save_path)

        # Create directory if it doesn't exsit.
        if os.path.isdir(self.model_path) is False:
            os.makedirs(self.model_path)

        # Create directory if it doesn't exsit.
        if os.path.isdir(self.video_path) is False:
            os.makedirs(self.video_path)

        # Create directory if it doesn't exsit.
        if os.path.isdir(self.video_train_path) is False:
            os.makedirs(self.video_train_path)

        # Create directory if it doesn't exsit.
        if os.path.isdir(self.video_eval_path) is False:
            os.makedirs(self.video_eval_path)

        self.eval_env = None
        self.writer = SummaryWriter(log_dir=self.save_path)

        self.settings = settings
        self.hyperparameters = hyperparameters
        self.rl_model_type = settings["rl_model_type"]
        self.shuffle_evaluation_idxs = settings["shuffle_evaluation_idxs"]
        self.model = model

    def on_step(self):
        if self.model.done:
            episode_length = self.model.trackr.episode_length[-1]
            mean_episode_length = np.int32(np.round(
                np.mean(self.model.trackr.episode_length),0)
                )
            episode_reward = np.round(self.model.trackr.episode_reward[-1],2)
            mean_episode_reward = np.round(
                np.mean(self.model.trackr.episode_reward),2)
            fps = np.round(self.model.trackr.episode_fps[-1],2)
            mean_fps = np.round(np.mean(self.model.trackr.episode_fps),2)

            if episode_reward > self.best_episode_reward:
                self.best_episode_reward =  episode_reward
                self.model.save(filename=os.path.join(self.model_path,
                                                      "best"))

            if mean_episode_reward > self.best_mean_episode_reward:
                self.best_mean_episode_reward = mean_episode_reward
                self.last_best_mean_timestep = self.model.num_timesteps
                self.model.save(filename=os.path.join(self.model_path,
                                                      "best_mean"))

            if (self.model.num_timesteps - self.last_best_mean_timestep) > self.model.early_stopping:
                self.model.exit = True

            self.writer.add_scalar(tag="rollout/mean_episode_length",
                                   scalar_value=mean_episode_length,
                                   global_step=self.model.num_timesteps)
            self.writer.add_scalar(tag="rollout/mean_episode_reward",
                                   scalar_value=mean_episode_reward,
                                   global_step=self.model.num_timesteps)
            self.writer.add_scalar(tag="rollout/fps",
                                   scalar_value=mean_fps,
                                   global_step=self.model.num_timesteps)

            mean_actor_loss = np.round(
                np.mean(self.model.trackr.critic_losses),2)
            mean_critic_loss = np.round(
                np.mean(self.model.trackr.actor_losses),2)

            self.writer.add_scalar(tag="training/mean_actor_loss",
                                   scalar_value=mean_actor_loss,
                                   global_step=self.model.num_timesteps)
            self.writer.add_scalar(tag="training/mean_critic_loss",
                                   scalar_value=mean_critic_loss,
                                   global_step=self.model.num_timesteps)

            if self.rl_model_type == "SAC":
                mean_critic_1_loss = np.round(
                    np.mean(self.model.trackr.critic_1_loss),2)
                mean_critic_2_loss = np.round(
                    np.mean(self.model.trackr.critic_2_loss),2)
                mean_entropy_loss = np.round(
                    np.mean(self.model.trackr.entropy_loss),2)
                entropy = np.round(self.model.trackr.entropy[-1],2)

                self.writer.add_scalar(tag="training/mean_critic_1_loss",
                                       scalar_value=mean_critic_1_loss,
                                       global_step=self.model.num_timesteps)
                self.writer.add_scalar(tag="training/mean_critic_2_loss",
                                       scalar_value=mean_critic_2_loss,
                                       global_step=self.model.num_timesteps)
                self.writer.add_scalar(tag="training/mean_entropy_loss",
                                       scalar_value=mean_entropy_loss,
                                       global_step=self.model.num_timesteps)
                self.writer.add_scalar(tag="training/entropy",
                                       scalar_value=entropy,
                                       global_step=self.model.num_timesteps)

            if self.training_log is not None:
                self.training_log.write(f"{self.model.episode_num},{self.model.num_timesteps},{episode_length},{np.round(self.model.trackr.episode_reward[-1],6)}\r\n")

            if self.verbose:
                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | {self.model.num_timesteps} | {episode_length}/{mean_episode_length} | {episode_reward}/{mean_episode_reward} | {fps}/{mean_fps}')

        if self.model.num_timesteps % self.eval_freq == 0:
            info = self.evaluate(
                model=None,
                tag=f"periodic_evals/training_dataset/{int(self.model.num_timesteps/1000)}k",
                training_dataset=True,
                quick_eval=True)
            self.writer.add_scalar(
                tag="periodic_evals/training_dataset/episode_reward",
                scalar_value=info[0],
                global_step=self.model.num_timesteps)
            self.writer.add_scalar(
                tag="periodic_evals/training_dataset/episode_length",
                scalar_value=info[1],
                global_step=self.model.num_timesteps)
            if self.verbose:
                mean_episode_length = np.int32(
                    np.round(np.mean(self.model.trackr.episode_length),0))
                mean_episode_reward = np.round(
                    np.mean(self.model.trackr.episode_reward),2)
                print(f" EVALUATION - TRAINING | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {self.model.num_timesteps} | {np.round(info[1],2)}/{mean_episode_length} | {np.round(info[0],2)}/{np.round(mean_episode_reward,2)}/{np.round(self.best_mean_episode_reward,2)}")

            info = self.evaluate(
                model=None,
                tag=f"periodic_evals/evaluation_dataset/{int(self.model.num_timesteps/1000)}k",
                training_dataset=False,
                quick_eval=True)
            self.writer.add_scalar(
                tag="periodic_evals/testing_dataset/episode_reward",
                scalar_value=info[0],
                global_step=self.model.num_timesteps)
            self.writer.add_scalar(
                tag="periodic_evals/testing_dataset/episode_length",
                scalar_value=info[1],
                global_step=self.model.num_timesteps)
            if self.verbose:
                mean_episode_length = np.int32(
                    np.round(np.mean(self.model.trackr.episode_length),0))
                mean_episode_reward = np.round(
                    np.mean(self.model.trackr.episode_reward),2)
                print(f" EVALUATION - TEST     | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {self.model.num_timesteps} | {np.round(info[1],2)}/{mean_episode_length} | {np.round(info[0],2)}/{np.round(mean_episode_reward,2)}/{np.round(self.best_mean_episode_reward,2)}")

    def on_training_start(self):
        print("Training started.")

        if self.training_log is not None:
            self.training_log = open(self.training_log, "w", encoding="utf-8")
            self.training_log.write(f"timestamp={datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')}\r\n")
            self.training_log.write("data begin\r\n")
            self.training_log.write("episode_number,timesteps,episode_length,accumulated_reward\r\n")

        self.initial_We = self.model.train_env.get_all_target_params()

        # log optimals
        if self.verbose > 2:
            print(f"Saving initial wirelength values, {self.initial_We}")
        f = open(self.optimals, "w", encoding="utf-8")
        f.write(f"intial={self.initial_We}\r\n")
        f.close()

        self.log_settings(self.settings,
                          initial_We=self.initial_We,
                          hyperparameters=self.hyperparameters,
                          model=self.model,
                          global_step=0)

    def on_training_end(self):
        print("Training finished")
        if self.training_log is not None:
            self.training_log.close()

        self.final_We = self.model.train_env.get_all_target_params()

        # log optimals
        if self.verbose > 2:
            print(f"Saving final wirelength values, {self.final_We}")
        f = open(self.optimals, "a", encoding="utf-8")
        f.write(f"final={self.final_We}\r\n")
        f.close()

        self.best_metrics = self.evaluate(model=os.path.join(
            self.model_path,"best"),
            tag="final_evals/training_dataset",
            training_dataset=True,
            quick_eval=False,
            periodic=False,
            long=True)

        self.best_mean_metrics = self.evaluate(model=os.path.join(
            self.model_path,"best_mean"),
            tag="final_evals/training_dataset",
            training_dataset=False,
            quick_eval=False,
            periodic=False,
            long=True)

        self.log_settings(self.settings,
                          initial_We=self.initial_We,
                          final_We=self.final_We,
                          hyperparameters=self.hyperparameters,
                          model=self.model,
                          global_step=0)

    def evaluate( self,
                  model: str,
                  tag: str,
                  training_dataset: bool = True,
                  quick_eval: bool=True,
                  periodic: bool= True,
                  long: bool= False,
                  verbose: int = 0 ):
        """
        :param model: Select model to use for evaluation. There are two options
         for selecting a model to use for evaluation: best and best_mean. The
         best model is the one that achieved the highest return during training
         while the best_mean model is the one that had the highest average
         return over 100 episodes.
        :type model: str
        :param tag: Tag in tensorboard
        :type tag: TYPE
        :param training_dataset: Select dataset to use for evaulation. When
         True, the training dataset is used, otherwise the evaluation dataset,
         defaults to True
        :type training_dataset: bool, optional
        :param quick_eval: When True saves the best layouts and creates an .mp4
         video file of the episode, defaults to True
        :type quick_eval: bool, optional
        :param periodic: Indicates whether the evaluation is periodic or final.
         Used to determine the logging directory and tensorboard tag for
         appropriately classifying the evaluations, defaults to True
        :type periodic: bool, optional
        :param long: Determines the evaluation duration. When True, the episode
         length is multiplied by multiplied by three, defaults to False
        :type long: bool, optional
        :param verbose: function verbosity, defaults to 0
        :type verbose: int, optional
        :return: List containing the average reward per episode and average
         steps per episode
        :rtype: TYPE

        """

        global best_reward

        if model is not None:
            self.model.load(model)

        file_best_hpwl_zero_overlap = "best_hpwl_zero_overlap"
        file_best_hpwl_10_overlap = "best_hpwl_10_overlap"
        file_best_hpwl_20_overlap = "best_hpwl_20_overlap"

        best_hpwl = 1E6
        best_hpwl_at_10_overlap = 1E6
        best_hpwl_at_20_overlap = 1E6

        params = copy.deepcopy(self.model.train_env.get_parameters())
        params.debug = True
        params.shuffle_idxs = self.shuffle_evaluation_idxs
        params.seed = 3142
        # Suppress logging of better expert paramater encounters
        params.log_dir = None
        # Evaluate on all evaluation layouts. In other words pcb_idx effects
        # on the training dataset.
        params.idx = -1
        if long is True:
            params.max_steps *= 3

        if training_dataset is True:
            params.pcb_file = params.training_pcb
            if periodic is True:
                video_tag = f"periodic_training_evaluations/{self.model.num_timesteps/1000}k"
            else:
                video_tag = "final_training_evaluation"
        else:
            params.pcb_file = params.evaluation_pcb
            if periodic is True:
                video_tag = f"periodic_testing_evaluations/{self.model.num_timesteps/1000}k"
            else:
                video_tag = "final_testing_evaluation"

        eval_env = environment(params)
        if training_dataset is True:
            target_params = self.model.train_env.get_target_params()
            for item in target_params:
                for i in range(len(eval_env.agents)):
                    if item["id"] == eval_env.agents[i].parameters.node.get_id():
                        eval_env.agents[i].We = item["We"]
                        eval_env.agents[i].HPWLe = item["HPWLe"]

        total_reward=0
        total_steps = 0
        for i in range(self.num_evaluations):
            if self.verbose > 1:
                print(f"Starting evaluation {i}/self.num_evaluations with pcb {eval_env.idx}.")

            if training_dataset is True:
                run_output_dir = self.video_train_path
            else:
                run_output_dir = self.video_eval_path

            if periodic is True:
                run_output_dir = os.path.join(
                    run_output_dir,
                    f"{int(self.model.num_timesteps/1000)}k/run_{i}")
            else:
                run_output_dir = os.path.join(run_output_dir, f"final/run_{i}")

            if os.path.isdir(run_output_dir) is False:
                os.makedirs(run_output_dir)

            evaluation_log = open(os.path.join(run_output_dir,"evaluation.log"), "w", encoding="utf-8")
            evaluation_log.write(f"timestamp={datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')}\r\n")
            evaluation_log.write("parameters begin\r\n")
            evaluation_log.write(self.model.train_env.parameters.to_text_string(prefix="\t"))
            evaluation_log.write("parameters end\r\n")

            best_hpwl = 1E6
            best_hpwl_at_10_overlap = 1E6
            best_hpwl_at_20_overlap = 1E6

            eval_env.reset()
            done = False
            episode_steps=0
            while not done:
                episode_steps += 1
                if self.rl_model_type == "TD3":
                    obs_vec = eval_env.step(model=self.model.actor,
                                            random=False,
                                            deterministic=True,
                                            rl_model_type=self.rl_model_type)
                else:
                    obs_vec = eval_env.step(model=self.model.policy,
                                            random=False,
                                            deterministic=True,
                                            rl_model_type=self.rl_model_type)
                step_reward=0

                if quick_eval is False: # save best layouts, log video
                    t = int(self.model.num_timesteps/1000)
                    hpwl = eval_env.calc_hpwl()

                    all_ol = []
                    for indiv_obs in obs_vec:
                        next_state_ol = indiv_obs[1][8:16]
                        ol_ratios = indiv_obs[-1]["ol_ratios"]
                        all_ol.append(np.sum(np.array(next_state_ol)*np.array(ol_ratios))*100)

                    if (hpwl < best_hpwl) and (np.max(all_ol) < 1):
                        best_hpwl = hpwl
                        filename = file_best_hpwl_zero_overlap + f"_{t}k_{i}.{self.num_evaluations-1}.{episode_steps}.pcb"
                        eval_env.write_current_pcb_file(path=run_output_dir, filename=filename)
                        evaluation_log.write(f"run={i}/{self.num_evaluations-1} @ episode_step={episode_steps} : Zero overlap best hpwl : hpwl={np.round(hpwl,4)}, overlap={np.round(np.sum(all_ol)/8,4)}\r\n")
                        evaluation_log.write(f"all_ol={all_ol}\r\n")

                        # Capture snapshot
                        snapshot_filename=f"{i}.{self.num_evaluations-1}.{episode_steps}"
                        eval_env.tracker.capture_snapshot(
                            fileName=os.path.join(run_output_dir,
                                                  snapshot_filename+".png")
                                                  )
                        eval_env.write_current_pcb_file(
                            path=run_output_dir,
                            filename= snapshot_filename+".pcb")
                        # overwrite best; unique filename for easier processing
                        # with automated tools
                        eval_env.write_current_pcb_file(
                            path=run_output_dir,
                            filename=file_best_hpwl_zero_overlap+".pcb")
                        if verbose == 1:
                            print(f"run={i}/{self.num_evaluations-1} @ episode_step={episode_steps} : Zero overlap best hpwl : hpwl={np.round(hpwl,4)}, overlap={np.round(np.sum(all_ol)/8,4)}")

                    if (hpwl < best_hpwl_at_10_overlap ) and (np.max(all_ol) <= 10):
                        best_hpwl_at_10_overlap = hpwl
                        filename = file_best_hpwl_10_overlap + f"_{int(t)}k_{i}.{self.num_evaluations-1}.{episode_steps}.pcb"
                        eval_env.write_current_pcb_file(
                            path=run_output_dir,
                            filename=filename)
                        evaluation_log.write(f"run={i}/{self.num_evaluations-1} @ episode_step={episode_steps} : 10% overlap best hpwl : hpwl={np.round(best_hpwl_at_10_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}\r\n")
                        evaluation_log.write(f"all_ol={all_ol}\r\n")

                        # Capture snapshot
                        snapshot_filename=f"{i}.{self.num_evaluations-1}.{episode_steps}"
                        eval_env.tracker.capture_snapshot(
                            fileName=os.path.join(run_output_dir,
                                                  snapshot_filename+".png")
                                                  )
                        eval_env.write_current_pcb_file(
                            path=run_output_dir,
                            filename= snapshot_filename+".pcb")
                        # overwrite best; unique filename for easier
                        # processing with automated tools
                        eval_env.write_current_pcb_file(
                            path=run_output_dir,
                            filename=file_best_hpwl_10_overlap+".pcb")
                        if verbose == 1:
                            print(f"run={i}/{self.num_evaluations-1} @ episode_step={episode_steps} : 10% overlap best hpwl : hpwl={np.round(best_hpwl_at_10_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}")

                    if (hpwl < best_hpwl_at_20_overlap) and (np.max(all_ol) <= 20):
                        best_hpwl_at_20_overlap = hpwl
                        filename = file_best_hpwl_20_overlap + f"_{int(t)}k_{i}.{self.num_evaluations-1}.{episode_steps}.pcb"
                        eval_env.write_current_pcb_file(
                            path=run_output_dir,
                            filename=filename)
                        evaluation_log.write(f"run={i}/{self.num_evaluations-1} @ episode_step={episode_steps} : 20% overlap best hpwl : hpwl={np.round(best_hpwl_at_20_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}\r\n")
                        evaluation_log.write(f"all_ol={all_ol}\r\n")

                        # Capture snapshot
                        snapshot_filename=f"{i}.{self.num_evaluations-1}.{episode_steps}"
                        eval_env.tracker.capture_snapshot(
                            fileName=os.path.join(run_output_dir,
                                                  snapshot_filename+".png")
                                                  )
                        eval_env.write_current_pcb_file(
                            path=run_output_dir,
                            filename=snapshot_filename+".pcb")
                        # overwrite best; unique filename for easier
                        # processing with automated tools
                        eval_env.write_current_pcb_file(
                            path=run_output_dir,
                            filename=file_best_hpwl_20_overlap+".pcb")
                        if verbose == 1:
                            print(f"run={i}/{self.num_evaluations-1} @ episode_step={episode_steps} : 20% overlap best hpwl : hpwl={np.round(best_hpwl_at_20_overlap,4)}, overlap={np.round(np.sum(all_ol)/8,4)}")

                for indiv_obs in obs_vec:
                    step_reward += indiv_obs[2]
                    if indiv_obs[4] is True:
                        done = True

                step_reward /= len(obs_vec)
                total_reward += step_reward
                total_steps += 1

            evaluation_log.write(f"eval_env episode {i} performed {episode_steps} in environment.\r\n")
            if verbose == 1:
                print(f"eval_env episode {i} performed {episode_steps} in environment.")
            eval_env.tracker.create_video(
                fileName=os.path.join(run_output_dir, f"{i}.mp4"),
                display_metrics=False)
            vids = eval_env.tracker.video_tensor()
            self.log_video(vids=vids, tag=video_tag, global_step=i)

            # eval_env.tracker.create_plot(fileName=os.path.join(video_path,
            #                                                    f'{i}.png'))
            eval_env.tracker.log_run_to_file(
                path=run_output_dir, filename=f"{i}.log",
                kicad_pcb=eval_env.g.get_kicad_pcb_file()
                )

            eval_env.tracker.reset()

        evaluation_log.close()

        return [total_reward / self.num_evaluations,
                total_steps / self.num_evaluations]

    def log_video(self,
                  vids,
                  tag:str="evaluation_run",
                  global_step:int=0):
        """
        Logs a 2D tensor of video frames to tensorboard
        :param vids: 2D video tensor
        :type vids: torch.tensor
        :param tag: Tensorboard tag, defaults to "evaluation_run"
        :type tag: str, optional
        :param global_step: Tensorboard global step number, defaults to 0
        :type global_step: int, optional
        :return: Nothing
        :rtype: TYPE

        """
        self.writer.add_video(tag=tag,
                              vid_tensor=vids,
                              global_step=global_step,
                              fps=30)
        self.writer.flush()

    def log_settings(self,
                     settings=None,
                     tag:str="settings",
                     initial_We=None,
                     final_We=None,
                     cmdline_args=None,
                     hyperparameters=None,
                     model=None,
                     global_step:int=0):
        """
        Log experiment settings to tensorboard
        :param settings: DESCRIPTION, defaults to None
        :type settings: TYPE, optional
        :param tag: Tensorboard tag, defaults to "settings"
        :type tag: str, optional
        :param initial_We: DESCRIPTION, defaults to None
        :type initial_We: TYPE, optional
        :param final_We: DESCRIPTION, defaults to None
        :type final_We: TYPE, optional
        :param cmdline_args: DESCRIPTION, defaults to None
        :type cmdline_args: TYPE, optional
        :param hyperparameters: DESCRIPTION, defaults to None
        :type hyperparameters: TYPE, optional
        :param model: DESCRIPTION, defaults to None
        :type model: TYPE, optional
        :param global_step: DESCRIPTION, defaults to 0
        :type global_step: int, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        s = str("")

        if cmdline_args is not None:
            s += str(cmdline_args)
            s += "<br><br>"

        if settings is not None:
            if type(settings) != dict:
                s += "Expected env_settings to be of type dictionary. Skipping.<br>"
            else:
                s += "<strong>env_settings</strong><br>"
                for key,value in settings.items():
                    s += f"{key} -> {value}<br>"

        s += "<br>"

        if initial_We is not None:
            s += "<strong>Initial Expert Parameters</strong><br>"
            for item in initial_We:
                s += f"{item}"
                s += "<br>"

        if final_We is not None:
            s += "<br>"
            s += "<strong>Final Expert Parameters</strong><br>"
            for item in final_We:
                s += f"{item}"
                s += "<br>"

        s += "<br>"

        if hyperparameters is not None:
            s += "<strong>hyperparameters</strong><br>"
            for key,value in hyperparameters.items():
                s += f"{key} -> {value}<br>"

        s += "<br>"

        if model is not None:
            s += f"<strong>{settings['rl_model_type']} Model Architecture</strong><br>"
            if settings["rl_model_type"] == "TD3":
                s += str(model.actor).replace("\n","<br>")
            else: # SAC
                s += str(model.policy).replace("\n","<br>")

            s += "<br><br>"
            s += str(model.critic).replace("\n","<br>")
            s += "<br><br>"
            s += str(model.critic_target).replace("\n","<br>")
            s += "<br><br>"
            s += "Activation function : "
            s += str(model.critic.activation_fn).replace("\n","<br>")
            s += "<br><br>"

        s += "<strong>Dependency Information</strong><br>"
        s += pcb.build_info_as_string().replace("\n","<br>")[4:-4]
        s += graph.build_info_as_string().replace("\n","<br>")

        self.writer.add_text(tag=tag, text_string=s, global_step=global_step)
        self.writer.flush()
