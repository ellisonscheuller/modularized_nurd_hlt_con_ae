import os
import random
import numpy as np
import torch
import torch.nn as nn
import wandb

from models.autoencoder import Autoencoder


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_standard_scaler(X, eps=1e-8):
    mu = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std)
    return mu, std

def transform_standard(X, mu, std):
    return (X - mu) / (std + 1e-8)


# per-sample MSE so we can save one score per event
class PerSampleMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
    def forward(self, recon, target):
        per_feat = self.mse(recon, target)
        return per_feat.mean(dim=1)


def zero_out_padding_np(X):
    X = X.copy()
    pad = (X == 0.0).all(axis=-1)
    X[pad] = 0.0
    return X

def flatten_np(x):
    n, nobj, fdim = x.shape
    return x.reshape(n, nobj * fdim)


def inference(ae, Xz_np, loss_fn, device, batch_size=4096):
    ae.eval()
    n = Xz_np.shape[0]
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, n, batch_size):
            i1 = min(i0 + batch_size, n)
            xb = torch.tensor(Xz_np[i0:i1], dtype=torch.float32, device=device)
            recon, _ = ae(xb)
            loss_b = loss_fn(recon, xb)
            out[i0:i1] = loss_b.detach().cpu().numpy()
    return out


def run_ae(config):
    set_seed(config.get("seed", 123))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    outdir = config.get("outdir", "./ae_outputs")
    os.makedirs(outdir, exist_ok=True)

    if config.get("use_wandb", True):
        wandb.init(
            project=config.get("wandb_project", "AE vs. Contrastive ABCD"),
            settings=wandb.Settings(_disable_stats=True),
            config=config,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        print(f"wandb run: {wandb.run.name}", flush=True)

    train_path = config["train_path"]
    test_path = config["test_path"]

    print("Loading train .pt:", train_path, flush=True)
    pt_train = torch.load(train_path, map_location="cpu")

    print("Loading test .pt:", test_path, flush=True)
    pt_test = torch.load(test_path, map_location="cpu")

    obj_train_t = pt_train["obj"]  # [N, 23, 4]
    obj_test_t = pt_test["obj"]
    eid_train = pt_train.get("eventid", None)
    eid_test = pt_test.get("eventid", None)

    print("train objects:", tuple(obj_train_t.shape), obj_train_t.dtype, flush=True)
    print("test  objects:", tuple(obj_test_t.shape), obj_test_t.dtype, flush=True)

    # only use first 4 features (pt, eta, phi, type_id)
    x_train = obj_train_t[:, :, :4].numpy().astype("float32")
    x_test = obj_test_t[:, :, :4].numpy().astype("float32")

    Xtr_raw = zero_out_padding_np(x_train)
    Xte_raw = zero_out_padding_np(x_test)

    X1_train_raw = flatten_np(Xtr_raw)
    X1_test_raw = flatten_np(Xte_raw)

    # standardize using train stats only
    mu1, std1 = fit_standard_scaler(X1_train_raw)
    X_train_z = transform_standard(X1_train_raw, mu1, std1)
    X_test_z = transform_standard(X1_test_raw, mu1, std1)

    feat = X_train_z.shape[1]
    reco_loss_fn = PerSampleMSE().to(device)

    ae_cfg = {
        "features": feat,
        "latent_dim": config["ae_latent"],
        "encoder_config": {"nodes": config["enc_nodes"]},
        "decoder_config": {"nodes": config["dec_nodes"] + [feat]},
        "alpha": config.get("alpha", 0.0),
    }
    ae = Autoencoder(ae_cfg).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=float(config["ae_lr"]))

    X1 = torch.tensor(X_train_z, dtype=torch.float32, device=device)

    print("Starting AE training...", flush=True)
    for epoch in range(int(config["epochs"])):
        perm = torch.randperm(len(X1), device=device)
        losses = []

        for i0 in range(0, len(X1), int(config["batch_size"])):
            idx = perm[i0 : i0 + int(config["batch_size"])]
            xb = X1[idx]

            recon, _ = ae(xb)
            loss = reco_loss_fn(recon, xb).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg = float(np.mean(losses))
        if config.get("use_wandb", True):
            wandb.log({"epoch": epoch, "RecoLoss_AE": avg})
        print(f"Epoch {epoch}: Reco Loss: {avg:.6f}", flush=True)

    ae_path = os.path.join(outdir, "ae_axis1.pth")
    torch.save(ae.state_dict(), ae_path)
    print("Saved AE weights:", ae_path, flush=True)
    if config.get("use_wandb", True):
        wandb.save(ae_path)

    ae_scores_train = inference(ae, X_train_z, reco_loss_fn, device, batch_size=int(config.get("infer_batch_size", 4096)))
    ae_scores_test = inference(ae, X_test_z, reco_loss_fn, device, batch_size=int(config.get("infer_batch_size", 4096)))

    train_scores_path = os.path.join(outdir, "ae_scores_bkg_train.pt")
    test_scores_path = os.path.join(outdir, "ae_scores_bkg_test.pt")

    torch.save(torch.from_numpy(ae_scores_train).float(), train_scores_path)
    torch.save(torch.from_numpy(ae_scores_test).float(), test_scores_path)
    print("Saved scores:", train_scores_path, flush=True)
    print("Saved scores:", test_scores_path, flush=True)

    # save scaler so we can reproduce the normalization later
    scaler_path = os.path.join(outdir, "scaler_axis1.pt")
    torch.save({"mu": torch.from_numpy(mu1), "std": torch.from_numpy(std1)}, scaler_path)
    print("Saved scaler:", scaler_path, flush=True)

    if eid_train is not None:
        eid_train_path = os.path.join(outdir, "eventid_train.pt")
        torch.save(eid_train if torch.is_tensor(eid_train) else torch.tensor(eid_train), eid_train_path)
        print("Saved event ids:", eid_train_path, flush=True)
        if config.get("use_wandb", True):
            wandb.save(eid_train_path)

    if eid_test is not None:
        eid_test_path = os.path.join(outdir, "eventid_test.pt")
        torch.save(eid_test if torch.is_tensor(eid_test) else torch.tensor(eid_test), eid_test_path)
        print("Saved event ids:", eid_test_path, flush=True)
        if config.get("use_wandb", True):
            wandb.save(eid_test_path)

    if config.get("use_wandb", True):
        wandb.save(train_scores_path)
        wandb.save(test_scores_path)
        wandb.save(scaler_path)
        wandb.finish()

    return {
        "ae": ae,
        "ae_path": ae_path,
        "outdir": outdir,
        "train_scores_path": train_scores_path,
        "test_scores_path": test_scores_path,
        "scaler_path": scaler_path,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="/eos/user/e/escheull/smcocktail_1M/hlt_smcocktail_train.pt")
    p.add_argument("--test_path", default="/eos/user/e/escheull/smcocktail_1M/hlt_smcocktail_test.pt")
    p.add_argument("--outdir", default="/eos/user/e/escheull/ae_outputs_axis1/")
    p.add_argument("--epochs", type=int, default=20)
    cli = p.parse_args()

    cfg_ae = {
        "train_path": cli.train_path,
        "test_path": cli.test_path,
        "outdir": cli.outdir,
        "ae_latent": 16,
        "enc_nodes": [512, 256],
        "dec_nodes": [256, 512],
        "alpha": 0.0,
        "ae_lr": 1e-3,
        "epochs": cli.epochs,
        "batch_size": 4096,
        "infer_batch_size": 4096,
        "seed": 123,
        "use_wandb": True,
        "wandb_project": "AE vs. Contrastive ABCD",
    }

    out = run_ae(cfg_ae)
    print("\nSaved AE files here:")
    print(out["ae_path"])
    print(out["train_scores_path"])
    print(out["test_scores_path"])
    print(out["scaler_path"])
