import torch
import torch.nn as nn

class ReweightScalar(nn.Module):
    """
    Input:  z (AE loss) shape [B] or [B,1]
    Output: (h, logits) where logits is [B, num_classes]
    """
    def __init__(self, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z):
        if isinstance(z, (tuple, list)):
            raise ValueError("ReweightScalar expects z tensor, got tuple/list")

        if z.dim() == 1:
            z = z.unsqueeze(1)
        z = z.float()
        h = z
        return h, self.net(z)
