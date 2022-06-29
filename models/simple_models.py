import torch
import torch.nn as nn
import sys

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