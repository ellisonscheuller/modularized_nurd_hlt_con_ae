import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy
from sklearn.model_selection import train_test_split
import argparse
import os

from models import Predictor, TransformerEncoder, LinearProbe, MaskingAgent
from dataloader import  PUPPIDataset as PUPPIDataset

@torch.no_grad()
def evaluate_linear_probe(encoder, linear_probe, val_loader, device):
    encoder.eval()
    linear_probe.eval()
    correct = 0
    total = 0
    for batch in val_loader:
        x = batch["features"].to(device)
        y = batch["label"].to(device)
        z, _ = encoder(x)
        z = z.mean(dim=1)
        logits = linear_probe(z)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total

def main(data_path, encoder_path):
    torch.cuda.empty_cache()
    os.makedirs("checkpoints_rl", exist_ok=True)
    data = torch.load(data_path)
    features = data[:, 0:100, 0:4]
    labels = data[:, 0, 4].long()
    num_classes = torch.unique(labels).numel()

    dataset = PUPPIDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = TransformerEncoder(dataset[0]["features"].shape[-1], embed_size=64, num_particles=100).to(device)
    
    encoder.load_state_dict(torch.load(encoder_path))
    #only fine-tune forward layers
    #for param in encoder.parameters():
    #    param.requires_grad = False
    #last_block = encoder.layers[-1]
    #for name, param in last_block.named_parameters():
    #    if any(k in name for k in ["ffn", "norm1", "norm2"]):
    #        param.requires_grad = True

    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    def momentum_update(encoder, target_encoder, m=0.995):
        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    predictor = Predictor(64, 2, 4, 100).to(device)
    masking_agent = MaskingAgent(13).to(device)
    linear_probe = LinearProbe(64, num_classes).to(device)

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(encoder_params + list(predictor.parameters()), lr=1e-3)
    agent_optimizer = torch.optim.Adam(masking_agent.parameters(), lr=1e-3)
    probe_optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3)

    for epoch in range(1, 51):
        encoder.train()
        predictor.train()
        masking_agent.train()
        total_jepa_loss = 0
        total_rl_loss = 0
        log_probs_all = []

        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)

            mask_probs = masking_agent(x)
            mask_dist = torch.distributions.Bernoulli(mask_probs)
            masks_context = mask_dist.sample()
            masks_target = 1.0 - masks_context
            log_probs = mask_dist.log_prob(masks_context).sum(dim=1)
            log_probs_all.append(log_probs)

            with torch.no_grad():
                all_latents, _ = target_encoder(x)
                target_masked = all_latents * masks_target.unsqueeze(-1)

            transformer_mask = (masks_context == 0).bool()
            context, _ = encoder(x, mask=transformer_mask)
            preds, _ = predictor(context, transformer_mask, masks_target)
            preds = preds * masks_target.unsqueeze(-1)

            jepa_loss = F.mse_loss(preds, target_masked)
            optimizer.zero_grad()
            jepa_loss.backward()
            optimizer.step()
            momentum_update(encoder, target_encoder)

            total_jepa_loss += jepa_loss.item()

        encoder.eval()
        linear_probe.train()
        for _ in range(10):
            for batch in train_loader:
                x = batch["features"].to(device)
                y = batch["label"].to(device)
                with torch.no_grad():
                    z, _ = encoder(x)
                    z = z.mean(dim=1)
                logits = linear_probe(z)
                loss = F.cross_entropy(logits, y)
                probe_optimizer.zero_grad()
                loss.backward()
                probe_optimizer.step()

        reward = evaluate_linear_probe(encoder, linear_probe, train_loader, device)
        reward_tensor = torch.tensor(reward, device=device)

        agent_optimizer.zero_grad()
        rl_loss = -torch.cat(log_probs_all).mean() * reward_tensor
        rl_loss.backward()
        agent_optimizer.step()
        total_rl_loss += rl_loss.item()

        print(f"Epoch {epoch}/50 | JEPA Loss: {total_jepa_loss:.4f} | RL Loss: {total_rl_loss:.4f} | Probe Acc: {reward:.4f}")
        save_path = f"./checkpoints_rl/encoder_epoch_{epoch}.pt"
        torch.save(encoder.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .pt data file")
    parser.add_argument("--encoder", type=str, required=True, help="Path to pretrained encoder .pth")
    args = parser.parse_args()
    main(args.data, args.encoder)