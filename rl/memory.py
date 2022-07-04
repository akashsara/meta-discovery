import random
from collections import namedtuple, deque
import numpy as np
import torch

DQNTransition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "action_mask")
)
PPOTransition = namedtuple(
    "Transition", ("state", "action", "action_mask", "log_prob", "return_", "advantage")
)


class SequentialMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(DQNTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PPOMemory:
    """
    The main distinctions between this and the DQN Memory are:
    1) We don't want to reuse items here. Each transition is used only once.
    2) We don't have a maximum memory size due to the above.
    """
    def __init__(self, batch_size):
        self.memory = []
        self.batch_size = batch_size
        self.batches = []
        self.last_used = 0

    def push(self, state, action, action_mask, log_prob, return_, advantage):
        """
        Note: All received items are torch.tensors of shape (N, D)
        Where N is shared and is the size of a particular episode
        And D is the dimensionality for that entity.
        So we need to push each transition into memory.
        """
        for s, a, a_m, l_p, r, adv in zip(state, action, action_mask, log_prob, return_, advantage):
            self.memory.append(PPOTransition(s, a, a_m, l_p, r, adv))

    def is_empty(self): 
        if len(self.states) == 0:
            return True
        return False

    def generate_batches(self):
        """
        We first flatten out the variables in our memory.
        Then we split it into batches.
        """
        n_items = len(self.memory)
        batches = np.arange(0, n_items, self.batch_size)
        indices = np.arange(n_items)
        np.random.shuffle(indices)
        # Store a list of batches to then return
        self.batches = [indices[i : i + self.batch_size] for i in batches]
        # Store the last used index in case we add more items before sampling
        self.last_used = n_items

    def get_num_batches(self):
        return len(self.batches)

    def sample(self):
        for batch in self.batches:
            yield [self.memory[i] for i in batch]

    def clear(self):
        self.memory = self.memory[self.last_used:]
        self.batches = []
        self.last_used = 0
