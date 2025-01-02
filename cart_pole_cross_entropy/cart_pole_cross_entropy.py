import typing as tt
import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, n_actions) 
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
