import os
os.environ["KERAS_BACKEND"] = "torch" # Needs to be set, some pruning layers as well as the quantizers are Keras
import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
keras.backend.set_image_data_format("channels_first")
import h5py
from models import *
from pquant import get_default_config, add_compression_layers, iterative_train, get_model_losses
from quantizers.fixed_point.fixed_point_ops import get_fixed_quantizer
from torch.optim.lr_scheduler import CosineAnnealingLR
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

# pruning_methods: "autosparse, cl, cs, dst, pdp, wanda"
pruning_method = "pdp"
config = get_default_config(pruning_method)
# Set target sparsity to 80% (20% of weights are non-zero). This parameter exists only for some pruning methods
config["pruning_parameters"]["sparsity"] = 0.8
print(config)

config['training_parameters']['epochs'] = 100
config['training_parameters']['pretraining_epochs'] = 20
config['training_parameters']['fine_tuning_epochs'] = 20

input_shape = (2048, 100, 13)
model = add_compression_layers(student, config, input_shape)
print(model)


file = h5py.File("/eos/project/c/cms-l1ml/public/hackathon/phase2_ml.h5", "r")
input = file['train'][:]
input_tensor = torch.tensor(input[:, :, 0:5], device=device)
input_labels = torch.tensor(input[:, 0, -1], dtype=torch.long, device=device)


norm_constants = compute_normalization_constants(input_tensor)
dataset = PUPPIDataset(input_tensor, input_labels, norm_constants, device=device, augment=False)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=0)

quantizer = get_fixed_quantizer(overflow_mode="SAT")

def train_student(model, trainloader, device, loss_func, epoch, optimizer, scheduler, *args, **kwargs):
    running_loss = 0.0
    num_batches = 0

    for x, mask, labels in trainloader:
        x, mask, labels = x.to(device), mask.to(device), labels.to(device)       
        input_student = quantizer(x, k=torch.tensor(1.), i=torch.tensor(0.), f=torch.tensor(7.))  
        encoder_mask = torch.cat([
            torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool),
            mask.bool()
        ], dim=1)
        
        delta_r = compute_delta_r(x)
        
        
        with torch.no_grad():
            target_latent = encoder(x, delta_r, encoder_mask)
       
        outputs = model(input_student)         
        optimizer.zero_grad()
        outputs = model(input_student)  
        loss = loss_func(outputs, target_latent)
        losses = get_model_losses(model, torch.tensor(0.).to(device))
        loss += losses
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")

    if scheduler is not None:
        scheduler.step()


def validate_student(model, testloader, device, loss_func, epoch, *args, **kwargs):
    correct = 0
    total = 10
    model.eval()
    with torch.no_grad():
        ratio = get_layer_keep_ratio(model)
        print(f'Accuracy: {100 * correct / total:.2f}%, remaining_weights: {ratio * 100:.2f}%')


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, 200)
loss_function = nn.MSELoss()

trained_student = iterative_train(model = model, 
                                config = config, 
                                train_func = train_student, 
                                valid_func = validate_student, 
                                trainloader = dataloader, 
                                testloader = dataloader, 
                                device = device, 
                                loss_func = loss_function,
                                optimizer = optimizer, 
                                scheduler = scheduler
                                )