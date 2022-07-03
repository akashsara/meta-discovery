import random
from collections import namedtuple, deque
import numpy as np
import torch

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "action_mask")
)


class SequentialMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.action_masks = []
        self.log_probs = []
        self.returns = []
        self.advantages = []
        self.batch_size = batch_size
        self.batches = []

    def push(self, state, action, action_mask, log_prob, return_, advantage):
        """
        Note: All received items are torch.tensors of shape (N, D)
        Where N is shared and is the size of a particular episode
        And D is the dimensionality for that entity
        """
        self.states.append(state)
        self.actions.append(action)
        self.action_masks.append(action_mask)
        self.log_probs.append(log_prob)
        self.returns.append(return_)
        self.advantages.append(advantage)

    def is_empty(self):
        if len(self.states) == 0:
            return True
        return False

    def generate_batches(self):
        """
        We first flatten out the variables in our memory.
        Then we split it into batches.
        """
        self.states = torch.cat(self.states, dim=0)
        self.actions = torch.cat(self.actions, dim=0)
        self.action_masks = torch.cat(self.action_masks, dim=0)
        self.log_probs = torch.cat(self.log_probs, dim=0)
        self.returns = torch.cat(self.returns, dim=0)
        self.advantages = torch.cat(self.advantages, dim=0)

        n_items = len(self.states)
        batches = np.arange(0, n_items, self.batch_size)
        indices = np.arange(n_items)
        np.random.shuffle(indices)
        self.batches = [indices[i : i + self.batch_size] for i in batches]

    def get_num_batches(self):
        return len(self.batches)

    def sample(self):
        for batch in self.batches:
            yield self.states[batch],  self.actions[batch], self.action_masks[batch], self.log_probs[batch], self.returns[batch], self.advantages[batch]

    def clear(self):
        self.states = []
        self.actions = []
        self.action_masks = []
        self.log_probs = []
        self.returns = []
        self.advantages = []
        self.batches = []
