import random
from collections import namedtuple, deque
import numpy as np

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