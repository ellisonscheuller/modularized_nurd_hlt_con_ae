import torch
import torch.nn as nn


class HLTCritic(nn.Module):

    def __init__(self, rep_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim + 2, hidden),  # +1 for y, +1 for z
            nn.ReLU(),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, rx, y, z):
        if y.dim() == 1:
            y = y.unsqueeze(1)
        y = y.float()

        if z.dim() == 1:
            z = z.unsqueeze(1)
        z = z.float()

        x = torch.cat([rx, y, z], dim=1)
        return self.net(x)
