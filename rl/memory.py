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
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py#L315
    """

    def __init__(
        self, batch_size, memory_size, gamma, gae_lambda, state_size, n_actions
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.memory_size = memory_size
        self.state_size = state_size
        self.n_actions = n_actions

        self.pos = 0
        self.states = np.zeros((self.memory_size, self.state_size), dtype=np.float32)
        self.actions = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.rewards = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.returns = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.episode_starts = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.values = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.advantages = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.action_masks = np.zeros(
            (self.memory_size, self.n_actions), dtype=np.float32
        )

    def clear(self):
        self.pos = 0
        self.states = np.zeros((self.memory_size, self.state_size), dtype=np.float32)
        self.actions = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.rewards = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.returns = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.episode_starts = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.values = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.advantages = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.action_masks = np.zeros(
            (self.memory_size, self.n_actions), dtype=np.float32
        )

    def push(self, state, action, reward, episode_start, value, log_probs, action_mask):
        """
        Note: All received items are torch.tensors of shape (N, D)
        Where N is shared and is the size of a particular episode
        And D is the dimensionality for that entity.
        So we need to push each transition into memory.
        """
        self.states[self.pos] = np.array(state).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_probs.clone().cpu().numpy()
        self.action_masks[self.pos] = np.array(action_mask).copy()
        self.pos += 1

    def compute_returns_and_advantage(self, last_value, done):
        # Ref: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py#L370
        last_value = last_value.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.memory_size)):
            if step == self.memory_size - 1:
                next_non_terminal = 1.0 - done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def sample(self):
        # Randomize batches
        indices = np.random.permutation(self.memory_size)
        # Return batches
        start_idx = 0
        while start_idx < self.memory_size:
            batch = indices[start_idx : start_idx + self.batch_size]
            start_idx += self.batch_size
            yield {
                "states": self.states[batch],
                "actions": self.actions[batch],
                "values": self.values[batch],
                "log_probs": self.log_probs[batch],
                "advantages": self.advantages[batch],
                "returns": self.returns[batch],
                "action_masks": self.action_masks[batch],
            }