import torch.nn as nn
import os
import torch

class MLP(nn.Module):
    def __init__(self, n_feats, num_classes):
        super().__init__()
        # architecture copied from simple_VAE.ipynb
        self.mlp = nn.Sequential(
            nn.Linear(n_feats, 80),
            nn.ReLU(),
            # nn.BatchNorm1d(80),
            nn.Linear(80, 80),
            nn.ReLU(),
            # nn.BatchNorm1d(80)
        )
        self.mlp_out = nn.Sequential(     
            nn.Linear(80, 80),
            nn.ReLU(),
            # nn.BatchNorm1d(80),
            nn.Linear(80, num_classes)
        )
    def forward(self, X):
        activations = self.mlp(X)
        return activations, self.mlp_out(activations)


class MLPCritic(nn.Module):
    def __init__(self, n_feats, num_classes):
        super().__init__()
        self.final_block = nn.Sequential(
            nn.Linear(81, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, rx, y):
        """
        Predict z from rx, y

        rx: [batch, 80]
        y: [batch, 1]

        """
        combined = torch.cat([rx, y], dim=1)  # [batch, 513]
        return self.final_block(combined)

def get_model_mlp(args):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(n_feats=12, num_classes=2)
    directory = "/misc/vlgscratch5/RanganathGroup/lily/physics_ood/nuisance-aware-ood-detection/checkpoints/{in_dataset}/{name}/{exp}/".format(
        in_dataset=args.in_dataset, name=args.project_name, exp=args.exp_name)
    model_file = os.path.join(directory, f"checkpoint_main.pth.tar")
    print(model_file)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device))["state_dict_model"])
    model.to(device)
    return model