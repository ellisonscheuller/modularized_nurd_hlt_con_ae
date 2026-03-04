import torch.nn as nn
import torch.nn.functional as F
from models.hlt_con.hlt_model import TransformerEncoder

class HLTContrastiveModel(nn.Module):
    """
    NuRD expects: forward(inputs) -> (activations, logits)
    Roy TransformerEncoder returns: latent vector
    We keep Roy unchanged and add a classifier head here.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_features: int = 7,
        num_tokens: int = 400,
        latent_dim: int = 6,
        embed_size: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        linear_dim=None,
        pairwise: bool = False,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_features=num_features,
            embed_size=embed_size,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            linear_dim=linear_dim,
            num_tokens=num_tokens,
            pre_processor=None,
            pairwise=pairwise,
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, inputs):
        x, mask = inputs
        z = self.encoder(x, mask=mask)          # [B, latent_dim]
        z = F.normalize(z, dim=1)               # L2-normalize for contrastive loss
        logits = self.classifier(z)             # [B, num_classes]
        return z, logits
