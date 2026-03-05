import torch
import torch.nn as nn
import gc
from torch.utils.data import DataLoader
import sys 
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.decomposition import PCA
from models import *
from dataloader import *
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.covariance import EmpiricalCovariance
import math


outdir = os.path.join(os.getcwd(), "evalPlot")
os.makedirs(outdir, exist_ok=True)


def main(data_path, encoder_path):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # device = "cpu"
    num_features = 7 #I changed it!
    embed_size = 128
    latent_dim = 6
    proj_dim = 12
    num_particles = 100
    lr = 1e-3

    encoder = TransformerEncoder(num_features, embed_size, latent_dim, num_particles).to(device).eval()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    encoder.eval()


    data = torch.load(data_path, map_location = device)[:, 0:100, :]
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    input_tensor = data[:, :, 0:7]
    input_labels = data[:, 0, -1].long()
    
    num_classes = int(input_labels.max().item()) + 1

    norm_constants = compute_normalization_constants(input_tensor)
    dataset = PUPPIDataset(input_tensor, input_labels, norm_constants, device=device, augment=False)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=0)


    batch_size = 1024
    datasets = {
        "label_{}".format(label): input_tensor[input_labels == label]
        for label in range(num_classes)
    }


    latents_all = {}
    norm_constants = compute_normalization_constants(input_tensor)

    for name, data in datasets.items():
        dataset = PUPPIDataset(data, labels=None, norm_constants=norm_constants, device=device, augment=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        torch.cuda.empty_cache()
        latents_list = []

        with torch.no_grad():
            for x, mask in dataloader:
                x = x.to(device)
                mask = mask.to(device)
                mask = torch.cat([
                torch.zeros(mask.size(0), 1, device=mask.device, dtype=mask.dtype), 
                    mask
                ], dim=1)
                x_phys = x.clone()
                eta_min = norm_constants["eta_min"].to(x.device)
                eta_max = norm_constants["eta_max"].to(x.device)
                x_phys[:, :, 1] = x[:, :, 1] * (eta_max - eta_min) + eta_min
                # convert phi [0,1] back to radians
                x_phys[:, :, 2] = x[:, :, 2] * (2 * math.pi) - math.pi
                delta_r = compute_delta_r(x_phys)
                latent = encoder(x, delta_r, mask)
                latents_list.append(latent)
                

        latents_all[name] = torch.cat(latents_list, dim=0)
        del dataset, dataloader, latents_list
        gc.collect()
        torch.cuda.empty_cache()

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA


    classes_to_plot = {
        'label_0': {'label': 'DY', 'color': 'blue', 'alpha': 0.1},
        'label_1': {'label': 'QCD', 'color': 'red', 'alpha': 0.1},
        'label_2': {'label': 'TT', 'color': 'green', 'alpha': 0.1},
        'label_3': {'label': 'WJet', 'color': 'orange', 'alpha': 0.1},
        'label_4': {'label': 'VBF', 'color': 'black', 'alpha': 0.1},
        # 'label_4': {'label': 'GluGluToHHTo4B', 'color': 'purple', 'alpha': 0.5},
        # 'label_5': {'label': 'Single Nu', 'color': 'black', 'alpha': 0.5},
        # 'label_6': {'label': 'SUEP', 'color': 'pink', 'alpha': 0.5},
    }


    embeddings_list = []
    slice_map = {}

    start_idx = 0
    for key, config in classes_to_plot.items():
        emb = latents_all[key].detach().cpu().numpy()
        embeddings_list.append(emb)
        N = emb.shape[0]
        slice_map[key] = (start_idx, start_idx + N)
        start_idx += N


    combined = np.vstack(embeddings_list)
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    plt.figure(figsize=(8, 6))
    for key, config in classes_to_plot.items():
        start, end = slice_map[key]
        emb_2d = combined_2d[start:end]
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                    s=5, alpha=config['alpha'], 
                    label=config['label'], 
                    color=config['color'])

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pca_latents.png"), dpi=300)
    plt.close()

    # === t-SNE ===
    tsne = TSNE(n_components=2, init="random", random_state=42, perplexity=50)
    combined_tsne = tsne.fit_transform(combined)

    plt.figure(figsize=(8, 6))
    for key, config in classes_to_plot.items():
        start, end = slice_map[key]
        emb_2d = combined_tsne[start:end]
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                    s=5, alpha=config['alpha'],
                    label=config['label'],
                    color=config['color'])
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tsne_latents.png"), dpi=300)
    plt.close()

    # === KDE plots per class (using first 2 PCA components for density) ===
    plt.figure(figsize=(12, 8))
    for i, (key, config) in enumerate(classes_to_plot.items()):
        start, end = slice_map[key]
        emb_2d = combined_2d[start:end]

        # KDE in 2D
        x, y = emb_2d[:, 0], emb_2d[:, 1]
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        xi, yi = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

        plt.contour(xi, yi, zi.reshape(xi.shape),
                    colors=config['color'],
                    alpha=0.7,
                    linewidths=1.0,
                    label=config['label'])

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("KDE Contours of Latents (PCA-reduced)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "kde_latents.png"), dpi=300)
    plt.close()

# ========================== Evaluate classifier
    all_latents = []
    all_labels = []
    for key in sorted(latents_all):
        latents_cpu = latents_all[key].detach().cpu()
        label_id = int(key.split("_")[1])
        all_latents.append(latents_cpu)
        all_labels.append(
            torch.full((latents_cpu.size(0),), label_id, dtype=torch.long)
        )

    latent_tensor = torch.cat(all_latents)
    label_tensor = torch.cat(all_labels)
    X = latent_tensor.numpy()
    y = label_tensor.numpy()

    class_label_map = {
        int(name.split("_")[1]): cfg["label"]
        for name, cfg in classes_to_plot.items()
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=512,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=512,
    )

    model = EvalMLP(input_dim=latent_tensor.shape[1], num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {running_loss / len(train_loader):.4f}")

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            all_logits.append(model(xb).cpu())
            all_targets.append(yb.cpu())

    logits = torch.cat(all_logits)
    probs = torch.softmax(logits, dim=1).numpy()
    y_pred = np.argmax(probs, axis=1)
    y_true = torch.cat(all_targets).numpy()

    target_names = [class_label_map[i] for i in sorted(np.unique(y_true))]
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=target_names))

    plt.figure(figsize=(10, 7))
    for class_id in np.unique(y_true):
        if class_id == 0:
            continue
        bin_true = (y_true == class_id).astype(int)
        bin_score = probs[:, class_id]
        fpr, tpr, _ = roc_curve(bin_true, bin_score)
        fpr_khz = fpr * (40 * 1e3) * (2760 / 3564)
        label_text = class_label_map.get(class_id, f"label_{class_id}")
        plt.plot(fpr_khz, tpr, label=f"{label_text} (AUC={auc(fpr, tpr):.2f})")

    plt.xscale("log")
    plt.xlabel("L1 Rate [kHz]")
    plt.ylabel("Signal Efficiency")
    plt.title("Supervised ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/supervised_rocs.png")
    plt.close()

    # Mahalanobis anomaly scoring
    bkg_latents = X[y == 1]
    cov = EmpiricalCovariance().fit(bkg_latents)
    mahalanobis_scores = cov.mahalanobis(X)
    anomaly_labels = (y != 1).astype(int)

    fpr, tpr, _ = roc_curve(anomaly_labels, mahalanobis_scores)
    fpr_khz = fpr * (40 * 1e3) * (2760 / 3564)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr_khz,
        tpr,
        label=f"Unsupervised Anomaly Score (AUC={auc(fpr, tpr):.2f})",
        color="black",
    )
    plt.xscale("log")
    plt.xlabel("L1 Rate [kHz]")
    plt.ylabel("Signal Efficiency")
    plt.title("Anomaly Detection ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/anomaly_rocs.png")
    plt.close()

    # Fisher / Mahalanobis ROC

    def fisher_pairwise_roc(X, y, A, B, out_png, label_map=None, eps=1e-6, title_prefix="Fisher ROC"):
        """
        Two-class post-hoc scoring:
        - Build within-class scatter Sw = cov(A) + cov(B) on TRAIN
        - Score on TEST: score = d_B(x) - d_A(x) using Mahalanobis under Sw
        - ROC: positives = class A
        Saves ROC figure and returns (auc_val, J, sqrtJ)
        """
        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        # collect class samples (TRAIN only)
        XA = X_train[y_train == A]
        XB = X_train[y_train == B]
        if XA.shape[0] < 3 or XB.shape[0] < 3:
            raise ValueError(f"Not enough samples: A={XA.shape[0]} B={XB.shape[0]}")

        # means
        mA = XA.mean(axis=0)
        mB = XB.mean(axis=0)

        # covariances (rowvar=False => columns are features)
        S1 = np.cov(XA, rowvar=False)
        S2 = np.cov(XB, rowvar=False)
        # within-class scatter + regularization
        Sw = S1 + S2 + eps * np.eye(S1.shape[0])
        Sw_inv = np.linalg.inv(Sw)

        # Fisher separation between centroids
        diff = (mA - mB)
        J = diff @ Sw_inv @ diff
        sqrtJ = math.sqrt(max(J, 0.0))

        # Mahalanobis distance under Sw
        def maha(X_, mean):
            d = X_ - mean
            return np.sqrt(np.sum((d @ Sw_inv) * d, axis=1))

        # score on TEST (higher => looks like A)
        dA = maha(X_test, mA)
        dB = maha(X_test, mB)
        score = dB - dA

        # ROC (positives = class A)
        y_bin = (y_test == A).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, score)
        auc_val = auc(fpr, tpr)

        # Convert FPR to L1 rate [kHz] like in your other plots
        fpr_khz = fpr * (40 * 1e3) * (2760 / 3564)

        # Labels for legend/title
        nameA = label_map.get(A, f"class_{A}") if label_map else f"class_{A}"
        nameB = label_map.get(B, f"class_{B}") if label_map else f"class_{B}"

        # Plot & save
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_khz, tpr, label=f"{nameA} vs {nameB} (AUC={auc_val:.3f})")
        plt.xscale("log")
        plt.xlabel("L1 Rate [kHz]")
        plt.ylabel(f"{nameA} Efficiency")
        plt.title(f"{title_prefix}: {nameA} vs {nameB}\nFisher J={J:.3f}, √J={sqrtJ:.3f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()

        print(f"[Fisher] {nameA} vs {nameB}: AUC={auc_val:.3f}, J={J:.3f}, sqrt(J)={sqrtJ:.3f}  → {out_png}")
        return auc_val, J, sqrtJ
    
    fisher_pairwise_roc(
        X, y, A=2, B=4,  # TT vs HH4B (adjust ids)
        out_png=os.path.join(outdir, "fisher_roc_TT_vs_HH4B.png"),
        # label_map=id_to_name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .pt test data")
    parser.add_argument("--encoder", required=True, help="Path to saved encoder .pt file")
    args = parser.parse_args()
    main(args.data, args.encoder)