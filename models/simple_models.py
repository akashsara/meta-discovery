import torch
import torch.nn as nn
import sys

class SimpleActorCriticModel(nn.Module):
    def __init__(self, n_actions):
        super(SimpleActorCriticModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
        )
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        features = self.model(state)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value

class SimpleModel(nn.Module):
    def __init__(self, n_actions):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, n_actions),
        )

    def forward(self, state):
        return self.model(state)