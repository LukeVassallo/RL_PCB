import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update, ReplayMemory
import time
import numpy as np

import tracker
import gym
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(torch.nn.Module):
    def __init__(self, num_inputs,
                 num_actions,
                 qf : list = [400, 300],
                 activation_fn: str = "relu"):

        super(QNetwork, self).__init__()

        if activation_fn == "relu":
            self.activation_fn = torch.nn.functional.relu
        elif activation_fn == "tanh":
            self.activation_fn = torch.nn.functional.tanh

        # Q1 architecture
        self.qf1 = torch.nn.Sequential()
        in_size = num_inputs + num_actions
        for layer_sz in qf:
            self.qf1.append(torch.nn.Linear(in_size, layer_sz))
            in_size = layer_sz
        self.qf1.append(torch.nn.Linear(in_size, 1))

		# Q2 architecture
        self.qf2 = torch.nn.Sequential()
        in_size = num_inputs + num_actions
        for layer_sz in qf:
            self.qf2.append(torch.nn.Linear(in_size, layer_sz))
            in_size = layer_sz
        self.qf2.append(torch.nn.Linear(in_size, 1))

        # self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = sa
        for i in range(len(self.qf1)-1):
            q1 = self.activation_fn(self.qf1[i](q1))
        q1 = self.qf1[-1](q1)

        q2 = sa
        for i in range(len(self.qf2)-1):
            q2 = self.activation_fn(self.qf2[i](q2))
        q2 = self.qf2[-1](q2)

        return q1, q2

class GaussianPolicy(torch.nn.Module):
    def __init__(self,
                 num_inputs,
                 num_actions,
                 pi : list = [400, 300],
                 activation_fn: str = "relu",
                 action_space=None,
                 device: str = "cuda"):
        super(GaussianPolicy, self).__init__()

        if device == "cuda":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.pi = torch.nn.Sequential()
        in_size = num_inputs
        for layer_sz in pi:
            self.pi.append(torch.nn.Linear(in_size, layer_sz))
            in_size = layer_sz

        self.mean_linear = torch.nn.Linear(in_size, num_actions)
        self.log_std_linear = torch.nn.Linear(in_size, num_actions)

        if activation_fn == "relu":
            self.activation_fn = torch.nn.functional.relu
        elif activation_fn == "tanh":
            self.activation_fn = torch.nn.functional.tanh

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = state
        for i in range(len(self.pi)):
            x = self.activation_fn(self.pi[i](x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # for reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.sample(state)
        else:
            _, _, action = self.sample(state)
        return action.detach().cpu().numpy()[0]

class SAC(object):
    def __init__(
            self,
            max_action,
            hyperparameters,
            train_env,
            device:str = "cpu",
            early_stopping:int = 100_000,
            # used when a training environment is not supplied
            state_dim:int = 23,
            action_dim:int = 3,
            verbose:int =0
            ):

        self.gamma = hyperparameters["gamma"]       # 0.99
        self.tau = hyperparameters["tau"]           # 0.005
        self.alpha = 0.2                            # args.alpha
        self.lr = hyperparameters["learning_rate"]  # 3e-4
        self.batch_size = hyperparameters["batch_size"]
        self.buffer_size = hyperparameters["buffer_size"]
        self.gradient_steps = 1
        self.policy_type = "Gaussian"
        self.target_update_interval = 1
        self.automatic_entropy_tuning = True

        if device == "cuda":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        if verbose > 1:
            print(f"Model SAC is configured to learn with device {self.device}")

        self.train_env = train_env

        if self.train_env is not None:
            state_dim = self.train_env.agents[0].get_observation_space_shape()
            action_dim = self.train_env.agents[0].action_space.shape[0]
        else:
            action_shape = (3,)
            action_space = gym.spaces.Box(
                low=np.array([0,0,0],dtype=np.float32),
                high=np.array([1,2*np.pi,1],dtype=np.float32))

        self.critic = QNetwork(state_dim,
                               action_dim,
                               hyperparameters["net_arch"]["qf"],
                               hyperparameters["activation_fn"]).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(state_dim,
                                      action_dim,
                                      hyperparameters["net_arch"]["qf"],
                                      hyperparameters["activation_fn"]).to(self.device)

        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning is True:
            if self.train_env is not None:
                self.target_entropy = -torch.prod(torch.Tensor(train_env.agents[0].action_space.shape).to(self.device)).item()
            else:
                self.target_entropy = -torch.prod(torch.Tensor(action_shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1,
                                         requires_grad=True,
                                         device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

        if self.train_env is not None:
            self.policy = GaussianPolicy(state_dim,
                                         action_dim,
                                         hyperparameters["net_arch"]["pi"],
                                         hyperparameters["activation_fn"],
                                         action_space=train_env.agents[0].action_space,
                                         device=device).to(self.device)
        else:
            self.policy = GaussianPolicy(state_dim,
                                         action_dim,
                                         hyperparameters["net_arch"]["pi"],
                                         hyperparameters["activation_fn"],
                                         action_space=action_space,
                                         device=device).to(self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        # Memory
        self.replay_buffer = ReplayMemory(
            capacity=hyperparameters["buffer_size"],
            device=self.device)
        self.trackr = tracker.tracker(avg_size=100,rl_policy_type="SAC")

        # Early stopping
        self.early_stopping = early_stopping
        self.exit = False

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size=batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def explore_for_expert_targets(self,
                                   reward_target_exploration_steps=25_000):
        if self.train_env is None:
            print("Model cannot explore because training envrionment is missing. Please reload model and supply a training envrionment.")
            return

        self.done = False
        for _ in range(reward_target_exploration_steps):
            obs_vec = self.train_env.step(self.policy,
                                          random=True,
                                          rl_model_type="SAC")

            for indiv_obs in obs_vec:
                if indiv_obs[4] is True:
                    self.done = True

            if self.done:
                self.train_env.reset()
                self.done = False
                self.train_env.tracker.reset()

        self.train_env.reset()
        self.done = False

    def learn(self,
              timesteps,
              callback,
              start_timesteps=25_000,
              incremental_replay_buffer = None):

        if self.train_env is None:
            print("Model cannot learn because training envrionment is missing. Please reload model and supply a training envrionment.")
            return

        next_update_at = self.buffer_size*2

        episode_reward = 0
        episode_timesteps = 0
        self.episode_num = 0

        callback.on_training_start()

        self.train_env.reset()
        self.done = False
        start_time = time.clock_gettime(time.CLOCK_REALTIME)

        episode_start_time = start_time

        # Training Loop
        updates = 0

        all_actor_losses = []
        all_critic_1_losses = []
        all_critic_2_losses = []
        all_entropy_losses = []

        alpha =0
        for t in range(1,int(timesteps)+1):
            self.num_timesteps = t
            episode_timesteps += 1

            if t < start_timesteps:
                obs_vec = self.train_env.step(model=self.policy,
                                              random=True,
                                              rl_model_type="SAC")
            else:
                obs_vec = self.train_env.step(model=self.policy,
                                              random=False,
                                              rl_model_type="SAC")

            all_rewards = []
            for indiv_obs in obs_vec:
                if indiv_obs[4] is True:
                    self.done = True
                all_rewards.append(indiv_obs[2])
                transition = (indiv_obs[0],
                              indiv_obs[3],
                              indiv_obs[1],
                              indiv_obs[2],
                              1. -indiv_obs[4])
                self.replay_buffer.add(*transition)

            episode_reward += float(np.mean(np.array(all_rewards)))

            if t >= start_timesteps:
                if len(self.replay_buffer) > self.batch_size:
                    # Number of updates per step in environment
                    for _ in range(self.gradient_steps):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.train(self.replay_buffer,
                                                                                                self.batch_size,
                                                                                                updates)
                        updates += 1

                        all_actor_losses.append(policy_loss)
                        all_critic_1_losses.append(critic_1_loss)
                        all_critic_2_losses.append(critic_2_loss)
                        all_entropy_losses.append(ent_loss)

            if self.done:
                episode_finish_time = time.clock_gettime(time.CLOCK_REALTIME)
                if t < start_timesteps or len(self.replay_buffer) <= self.batch_size:
                    self.trackr.append(actor_loss=0,
                           critic_loss=0,
                           episode_reward=episode_reward,
                           episode_length = episode_timesteps,
                           episode_fps = episode_timesteps / (episode_finish_time - episode_start_time),
                           critic_1_loss=0,
                           critic_2_loss=0,
                           entropy_loss=0,
                           entropy=0)
                else:
                    self.trackr.append(actor_loss=np.mean(all_actor_losses),
                           critic_loss=np.mean(all_critic_1_losses+all_critic_2_losses),
                           episode_reward=episode_reward,
                           episode_length = episode_timesteps,
                           episode_fps = episode_timesteps / (episode_finish_time - episode_start_time),
                           critic_1_loss=np.mean(all_critic_1_losses),
                           critic_2_loss=np.mean(all_critic_2_losses),
                           entropy_loss=np.mean(all_entropy_losses),
                           entropy=alpha)

            callback.on_step()
            if self.done:
                self.train_env.reset()
                self.done = False
                episode_reward = 0
                episode_timesteps = 0
                self.episode_num += 1
                self.train_env.tracker.reset()
                episode_start_time = time.clock_gettime(time.CLOCK_REALTIME)

                all_actor_losses = []
                all_critic_1_losses = []
                all_critic_2_losses = []
                all_entropy_losses = []

			# Early stopping
            if self.exit is True:
                print(f"Early stopping mechanism triggered at timestep={self.num_timesteps} after {self.early_stopping} steps without improvement ... Learning terminated.")
                break

            if incremental_replay_buffer is not None:
                if t >= next_update_at:
                    if incremental_replay_buffer == "double":
                        self.buffer_size *= 2
                        next_update_at += self.buffer_size * 2
                    elif incremental_replay_buffer == "triple":
                        self.buffer_size *= 3
                        next_update_at += self.buffer_size
                    elif incremental_replay_buffer == "quadruple":
                        self.buffer_size *= 4
                        next_update_at += self.buffer_size

                    old_replay_buffer = self.replay_buffer
                    self.replay_buffer = ReplayMemory(self.buffer_size,
                                                      device=self.device)
                    self.replay_buffer.add_content_of(old_replay_buffer)

                    print(f"Updated replay buffer at timestep {t}; replay_buffer_size={self.buffer_size}, len={self.replay_buffer.__len__()} next_update_at={next_update_at}")

        callback.on_training_end()

    # Save model parameters
    def save(self, filename):
        torch.save({"policy_state_dict": self.policy.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "critic_target_state_dict": self.critic_target.state_dict(),
                    "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                    "policy_optimizer_state_dict": self.policy_optim.state_dict()}, filename)

    # Load model parameters
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(
            checkpoint["critic_target_state_dict"])
        self.critic_optim.load_state_dict(
            checkpoint["critic_optimizer_state_dict"])
        self.policy_optim.load_state_dict(
            checkpoint["policy_optimizer_state_dict"])
