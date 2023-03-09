import math
import torch
import numpy as np

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

from collections import namedtuple
import random 
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        reshaped_args = []
        for arg in args:
            reshaped_args.append(np.reshape(arg, (1, -1)))

        self.memory[self.position] = Transition(*reshaped_args)
        self.position = (self.position + 1) % self.capacity

    def add_content_of(self, other):
        """
        Adds the content of another replay buffer to this replay buffer
        :param other: another replay buffer
        """
        latest_trans = other.get_latest(self.capacity)
        for transition in latest_trans:
            self.add(*transition)

    def get_latest(self, latest):
        """
        Returns the latest element from the other buffer with the most recent ones at the end of the returned list
        :param other: another replay buffer
        :param latest: the number of latest elements to return
        :return: a list with the latest elements
        """
        if self.capacity < latest:
            latest_trans = self.memory[self.position:].copy() + self.memory[:self.position].copy()
        elif len(self.memory) < self.capacity:
            latest_trans = self.memory[-latest:].copy()
        elif self.position >= latest:
            latest_trans = self.memory[:self.position][-latest:].copy()
        else:
            latest_trans = self.memory[-latest+self.position:].copy() + self.memory[:self.position].copy()
        return latest_trans

    def add_latest_from(self, other, latest):
        """
        Adds the latest samples from the other buffer to this buffer
        :param other: another replay buffer
        :param latest: the number of elements to add
        """
        latest_trans = other.get_latest(latest)
        for transition in latest_trans:
            self.add(*transition)

    def shuffle(self):
        random.shuffle(self.memory)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.FloatTensor(np.concatenate(batch.state)).to(self.device)
        action = torch.FloatTensor(np.concatenate(batch.action)).to(self.device)
        next_state = torch.FloatTensor(np.concatenate(batch.next_state)).to(self.device)
        reward = torch.FloatTensor(np.concatenate(batch.reward)).to(self.device)
        done = torch.FloatTensor(np.concatenate(batch.done)).to(self.device)
        return state, action, next_state, reward, done

    def sample_from_latest(self, batch_size, latest):
        latest_trans = self.get_latest(latest)
        transitions = random.sample(latest_trans, batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.FloatTensor(np.concatenate(batch.state)).to(self.device)
        action = torch.FloatTensor(np.concatenate(batch.action)).to(self.device)
        next_state = torch.FloatTensor(np.concatenate(batch.next_state)).to(self.device)
        reward = torch.FloatTensor(np.concatenate(batch.reward)).to(self.device)
        done = torch.FloatTensor(np.concatenate(batch.done)).to(self.device)
        return state, action, next_state, reward, done

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = []
        self.position = 0    
