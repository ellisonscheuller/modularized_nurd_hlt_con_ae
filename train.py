import torch
import torch.nn as nn
import datetime
import os
import logging
import argparse
import importlib
import wandb
from embedding.models import TransformerEncoder, Projector
from embedding.training import make_train_val_split, build_train_val_loaders, train_epoch, validate_epoch, EarlyStopping, cosine_schedule_with_warmup, cosine_constrastive_schedule
from embedding.utils.data_utils import compute_normalization_constants
from embedding.utils.cfg_handler import train_config, data_config
from embedding.utils.data_utils import compute_class_weights, load_data

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("JEPA") 
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/training_{timestamp}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def main(data_path: str, cfg: train_config, cfg_data: data_config, test_mode: bool = False):

    run = wandb.init(
        project="embedding_hlt", 
        config = {
            **cfg.get_entire_cfg(),
            **cfg_data.get_entire_cfg()
        },
    )
    is_sweep = cfg.is_sweep()
    if is_sweep:
        logger.info("Running in sweep mode with wandb.config overrides.")
    if not is_sweep:
        run.name = f"{cfg.get_model_name()}_{timestamp}"
        logger.info(f"Running in standard mode using config file values. Run name: {run.name}")

    num_epochs = cfg.hp("num_epochs", 400 if not is_sweep else 50)
    patience = cfg.hp("early_stopping_patience", 100 if not is_sweep else 20)
    val_split = cfg.get_trdata_cfg("val_split", 0.1)
    pairwise = cfg.get_trdata_cfg("pairwise", False)
    class_weights_setting = cfg_data.get("class_weights", None)
    pfcands = cfg_data.get("pfcands", True)
    contrast_loss = cfg.hp("contrast_loss", "InfoNCELoss")
    preproc_type = cfg.get_trdata_cfg("preproc_type", "PFPreProcessor")
    mixed_prec = cfg.hp("mixed_prec", False)

    # Sweepable
    num_heads = cfg.hp("num_heads", 8)
    num_layers = cfg.hp("num_layers", 4)
    embed_size = cfg.hp("embed_size", 128)
    latent_dim = cfg.hp("latent_dim", 6)
    proj_dim = cfg.hp("proj_dim", 12)
    linear_dim = cfg.hp("linear_dim", None)
    contrast_temp = cfg.hp("contrast_temp", 0.07)
    contrastive_weight = cfg.hp("contrastive_weight", 0.05) # Min
    contrastive_max = cfg.hp("contrastive_max", None) # Max for schedule, None for fixed
    contrastive_warmup = cfg.hp("contrastive_warmup", 0.05)
    lr = cfg.hp("lr", 1e-3)
    lr_min = cfg.hp("lr_min", 0.0)
    lr_warmup = cfg.hp("lr_warmup", 0.05)
    batch_size = cfg.hp("batch_size", 256)

    logger.info("Scaler for mixed precision training: {}".format(mixed_prec))
    scaler = torch.cuda.amp.GradScaler(enabled=((device=="cuda") and mixed_prec))

    # Load and split
    if test_mode:
        num_events = int(0.10 * cfg_data.get("nevents_per_class") * cfg_data.get_file_label_map().__len__())
        logger.info("Test mode enabled: using only 10% of the data for training and validation.")
        logger.info(f"Number of events: {num_events} (10% of total)")

    feature_block, label_block = load_data(
        data_path, 
        map_location="cpu", # load on CPU and move to GPU in DataLoader to avoid GPU memory issues
        max_events=num_events if test_mode else -1,
    )
    X_tr, y_tr, X_val, y_val, _, _ = make_train_val_split(feature_block, label_block, val_size=val_split)
    assert X_tr.device.type == "cpu"
    assert y_tr.device.type == "cpu"

    # Log num classes
    num_classes = int(label_block.max().item()) + 1
    class_count_tr = torch.bincount(y_tr, minlength=num_classes)
    logger.info("Class counts in training set:")
    for i in range(num_classes):
        logger.info(f"  Class {i}: {class_count_tr[i].item()} events")
    class_count_val = torch.bincount(y_val, minlength=num_classes)
    logger.info("Class counts in validation set:")
    for i in range(num_classes):
        logger.info(f"  Class {i}: {class_count_val[i].item()} events")

    # Build loaders (use train stats for both)
    norm_constants = compute_normalization_constants(X_tr) if not pfcands else {}
    train_loader, val_loader = build_train_val_loaders(
        X_tr, y_tr, X_val, y_val, device=device, batch_size=batch_size, pfcands=pfcands
    )

    preproc_class = getattr(importlib.import_module("embedding.preprocs"), preproc_type)
    preproc = preproc_class(norm_constants).to(device)
    
    encoder = TransformerEncoder(
        num_features=preproc.num_features,
        embed_size=embed_size, 
        latent_dim=latent_dim, 
        num_heads=num_heads,
        num_layers=num_layers,
        linear_dim=linear_dim, 
        num_tokens=feature_block.size(1) if linear_dim is not None else None,
        pairwise=pairwise,
        pre_processor=preproc
    ).to(device).train()
    projector = Projector(latent_dim, proj_dim, hidden_dim=(proj_dim*4)).to(device).train()
    classifier = nn.Linear(proj_dim, num_classes).to(device).train()

    class_weights = compute_class_weights(label_block, setting=class_weights_setting).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    contrast_loss_class = getattr(importlib.import_module("embedding.loss"), contrast_loss)
    criterion = contrast_loss_class(temperature=contrast_temp)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + 
        list(projector.parameters()) + 
        list(classifier.parameters()),
        lr=lr
    )

    # Scheduler based on TRAIN steps
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(lr_warmup * total_steps)
    scheduler = cosine_schedule_with_warmup(
        optimizer, 
        warmup_steps, 
        total_steps,
        lr=lr, # max present => scheduled, otherwise use lr_max as fixed min
        lr_min=lr_min
    )

    # Scheduler for contrastive weight
    if contrastive_max is not None:
        contrastive_warmup_steps = int(contrastive_warmup * total_steps)
        contrastive_schedule = cosine_constrastive_schedule(
            weight_min = contrastive_weight,
            weight_max = contrastive_max,
            warmup_steps = contrastive_warmup_steps,
            total_steps = total_steps
        )

    best_val = float("inf")
    es = EarlyStopping(patience=patience, mode="min", min_delta=0.0)

    model_path = os.path.join(os.getcwd(), "checkpoints", f"{cfg.get_model_name()}_encoder_{timestamp}.pth")

    logger.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        tr = train_epoch(
            encoder, 
            projector, 
            classifier,
            ce_loss_fn, 
            criterion,
            train_loader, 
            norm_constants, 
            device,
            optimizer, 
            scheduler=scheduler, 
            contrastive_weight=contrastive_weight if contrastive_max is None else contrastive_schedule,
            pairwise=pairwise, 
            num_classes=num_classes,
            scaler=scaler
        )
        va = validate_epoch(
            encoder, 
            projector, 
            classifier,
            ce_loss_fn, 
            criterion,
            val_loader, 
            norm_constants, 
            device,
            contrastive_weight=contrastive_weight if contrastive_max is None else contrastive_schedule,
            pairwise=pairwise, 
            num_classes=num_classes
        )

        log_str = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train: loss {tr['loss']:.6f}, Contrast {tr['contrast']:.6f}, CrossEntropy {tr['ce']:.6f}, acc {tr['acc']:.4f} | "
            f"Val:   loss {va['loss']:.6f}, Contrast {va['contrast']:.6f}, CrossEntropy {va['ce']:.6f}, acc {va['acc']:.4f}"
        )
        logger.info(log_str)

        # save best on validation loss 
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "projector": projector.state_dict(),
                    "classifier": classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "norm_constants": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in norm_constants.items()},
                }, model_path
            )
            logger.info(f"Saved best encoder to: {model_path}")
        
        run.log(
            {
                # eval metrics
                "Train Loss": tr["loss"],
                "Train Constrastive": tr["contrast"],
                "Train CrossEntropy": tr["ce"],
                "Train Accuracy": tr["acc"],
                "Val Loss": va["loss"],
                "Val Contrastive": va["contrast"],
                "Val CrossEntropy": va["ce"],
                "Val Accuracy": va["acc"],
                # to track change in hps
                "Learning Rate": scheduler.get_last_lr()[0],
                "Contrastive Weight": contrastive_weight if contrastive_max is None else contrastive_schedule.get()
            }, step=epoch
        )

        # log lr and contrast weight
        current_lr = scheduler.get_last_lr()[0]
        run.log({"Learning Rate": current_lr}, step=epoch)
        if contrastive_max is not None:
            current_contrastive_weight = contrastive_schedule.get()
            run.log({"Contrastive Weight": current_contrastive_weight}, step=epoch)

        if es.step(va["loss"]):
            logger.info("Early stopping triggered.")
            break
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True, help="Path to the data config .yaml file") 
    parser.add_argument("--train_cfg", required=True, help="Path to the training config .yaml file")
    parser.add_argument("--data", required=True, help="Path to the input .pt file")
    parser.add_argument("--test_mode", action="store_true", help="If set, runs training with only 10 percent of the data.")
    args = parser.parse_args()

    tr_cfg = train_config(args.train_cfg)
    data_cfg = data_config(args.data_cfg)
    
    logger.info(f"Using train config file: {args.train_cfg}")
    logger.info(f"Entire train config: {tr_cfg.get_entire_cfg()}")
    
    logger.info(f"Using data processing config file: {args.data_cfg}")
    logger.info(f"Entire data processing config: {data_cfg.get_entire_cfg()}")

    main(args.data, tr_cfg, data_cfg, test_mode=args.test_mode)