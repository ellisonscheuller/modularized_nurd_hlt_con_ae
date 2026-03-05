import torch
import torch.nn as nn
import gc
from torch.utils.data import DataLoader
import sys 
import os
import h5py
import datetime
from models import *
from dataloader import *


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


num_features = 13
embed_size = 128
latent_dim = 6
proj_dim = 12
num_particles = 100
lr = 1e-3

encoder = TransformerEncoder(num_features, embed_size, latent_dim, num_particles).to(device).eval()

state_dict = torch.load("")
encoder.load_state_dict(state_dict)


encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

NUM_PARTICLES = 100
NUM_FEATURES = 13
NUM_CLASSES = 6
NUM_BLOCKS = 4
TOKEN_MLP_DIM = 128
CHANNEL_MLP_DIM = 256

student = MLPMixer(
    num_particles=NUM_PARTICLES,
    num_features=NUM_FEATURES,
    num_classes=NUM_CLASSES,
    num_blocks=NUM_BLOCKS,
    token_mlp_dim=TOKEN_MLP_DIM,
    channel_mlp_dim=CHANNEL_MLP_DIM
).to(device)

optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
mse_loss = nn.MSELoss()

# Data
file = h5py.File("/eos/project/c/cms-l1ml/public/hackathon/phase2_ml.h5", "r")
input = file['train'][:]
input_tensor = torch.tensor(input[:, :, 0:5], device=device)
input_labels = torch.tensor(input[:, 0, -1], dtype=torch.long, device=device)


norm_constants = compute_normalization_constants(input_tensor)
dataset = PUPPIDataset(input_tensor, input_labels, norm_constants, device=device, augment=False)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=0)


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

log_path = os.path.join(os.getcwd(), f"training_log_{timestamp}.txt")
model_path = os.path.join(os.getcwd(), f"student-fine-tuned_{timestamp}.pth")

# Train loop
num_epochs = 400
for epoch in range(num_epochs):
    total_loss = 0.0
    total_samples = 0

    for x, mask, labels in dataloader:
        encoder_mask = torch.cat([
            torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool),
            mask.bool()
        ], dim=1)
        
        delta_r = compute_delta_r(x)
        
        with torch.no_grad():
            target_latent = encoder(x, delta_r, encoder_mask)
        
        student_latent = student(x)
        loss = mse_loss(student_latent, target_latent)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        del x, mask, labels, target_latent, student_latent
        torch.cuda.empty_cache()
        gc.collect()
    

    avg_loss = total_loss / total_samples
    torch.save(encoder.state_dict(), model_path)
    with open(log_path, "a") as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}, Distill MSE Loss: {avg_loss:.6f}" + "\n")
    print(f"Epoch {epoch+1}/{num_epochs}, Distill MSE Loss: {avg_loss:.6f}")

sys.exit()
