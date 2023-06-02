import torch
import numpy as np
from collections import namedtuple
import random

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
            )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """
    A replay memory buffer used in reinforcement learning algorithms to store\
          and sample transitions.

    Args:
        capacity (int): The maximum capacity of the replay memory.
        device (str): The device to store the tensors (e.g., 'cpu', 'cuda').

    Attributes:
        capacity (int): The maximum capacity of the replay memory.
        device (str): The device to store the tensors.
        memory (list): A list to store the transitions.
        position (int): The current position in the memory buffer.

    Methods:
        add(*args): Saves a transition to the replay memory.
        add_content_of(other): Adds the content of another replay buffer to\
              this replay buffer.
        get_latest(latest): Returns the latest elements from the replay memory.
        add_latest_from(other, latest): Adds the latest samples from another\
             buffer to this buffer.
        shuffle(): Shuffles the transitions in the replay memory.
        sample(batch_size): Samples a batch of transitions from the replay\
              memory.
        sample_from_latest(batch_size, latest): Samples a batch of transitions\
              from the latest elements.
        __len__(): Returns the number of transitions stored in the replay\
              memory.
        reset(): Resets the replay memory by clearing all stored transitions.
    """

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
        Adds the content of another replay buffer to this replay buffer.

        Args:
            other (ReplayMemory): Another replay buffer.
        """
        latest_trans = other.get_latest(self.capacity)
        for transition in latest_trans:
            self.add(*transition)

    def get_latest(self, latest):
        """
        Returns the latest elements from the replay memory.

        Args:
            latest (int): The number of latest elements to return.

        Returns:
            list: A list containing the latest elements.
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
        Adds the latest samples from another buffer to this buffer.

        Args:
            other (ReplayMemory): Another replay buffer.
            latest (int): The number of elements to add.
        """
        latest_trans = other.get_latest(latest)
        for transition in latest_trans:
            self.add(*transition)

    def shuffle(self):
        """Shuffles the transitions in the replay memory."""
        random.shuffle(self.memory)

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the replay memory.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            tuple: A tuple containing the sampled tensors\
                  (state, action, next_state, reward, done).
        """
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.FloatTensor(np.concatenate(batch.state)).to(self.device)
        action = torch.FloatTensor(np.concatenate(batch.action)).to(self.device)
        next_state = torch.FloatTensor(np.concatenate(batch.next_state)).to(self.device)
        reward = torch.FloatTensor(np.concatenate(batch.reward)).to(self.device)
        done = torch.FloatTensor(np.concatenate(batch.done)).to(self.device)
        return state, action, next_state, reward, done

    def sample_from_latest(self, batch_size, latest):
        """
        Samples a batch of transitions from the latest elements in the\
              replay memory.

        Args:
            batch_size (int): The size of the batch to sample.
            latest (int): The number of latest elements to consider.

        Returns:
            tuple: A tuple containing the sampled tensors\
                  (state, action, next_state, reward, done).
        """
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
