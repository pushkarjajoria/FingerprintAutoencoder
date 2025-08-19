import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm import tqdm
from data.prepare_data import load_and_prepare
from models.autoencoder import FingerprintAutoencoder
import config


def train(compression, csv_path, base_output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        print(f"GPU Information: {torch.cuda.get_device_name(device_id)}")
    else:
        print("CUDA is not available. Using CPU.")

    output_dir = base_output_dir + f"/compression_{int(compression*100)}"
    os.makedirs(output_dir, exist_ok=True)
    cfg = config
    latent_dim = int(cfg.FINGERPRINT_DIM * compression)

    # Initialize wandb run
    wandb_run_name = f"Fingerprint_AE_{int(compression * 100)}_percent"
    run = wandb.init(
        project="morganfps-autoencoder",
        name=wandb_run_name,
        config={
            "compression": compression,
            "latent_dim": latent_dim,
            "fp_dim": cfg.FINGERPRINT_DIM,
            "batch_size": cfg.BATCH_SIZE,
            "epochs": cfg.EPOCHS,
            "learning_rate": cfg.LR,
            "early_stopping_patience": cfg.EARLY_STOPPING_PATIENCE
        }
    )
    config_w = run.config

    # Load data
    cache_file = cfg.CACHE_DATASET_PATH
    X_train, X_val = load_and_prepare(csv_path, cache_file)
    train_ds = TensorDataset(X_train)
    val_ds = TensorDataset(X_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)

    model = FingerprintAutoencoder(input_dim=cfg.FINGERPRINT_DIM, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.BCELoss()

    run.watch(model, log="all")

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    for epoch in tqdm(range(cfg.EPOCHS)):
        model.train()
        total_loss = 0.
        for step, (x,) in enumerate(train_loader):
            x = x.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % 10 == 0:
                run.log({"train/batch_loss": loss.item(), "epoch": epoch + 1})

        avg_train = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                val_loss += criterion(model(x), x).item()
        avg_val = val_loss / len(val_loader)

        run.log({
            "train/epoch_loss": avg_train,
            "val/epoch_loss": avg_val,
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        # Early stopping logic
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model so far
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{cfg.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final (best) model
    fname = os.path.join(output_dir, f"ae_final.pth")
    torch.save(best_model_state, fname)

    # Log to wandb
    artifact = wandb.Artifact(
        name=f"ae_comp{int(compression*100)}_final",
        type="model",
        metadata={
            "final_epoch": epoch + 1,
            "best_epoch": best_epoch,
            "compression": compression,
            "early_stopped": True if patience_counter >= cfg.EARLY_STOPPING_PATIENCE else False
        }
    )
    artifact.add_file(fname)
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression", type=float, required=True,
                        choices=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                        help="latent as fraction of fingerprint dimension")
    parser.add_argument("--csv", type=str, required=True, help="path to SMILES csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="directory to save model checkpoints")

    args = parser.parse_args()

    # Pass output_dir to train
    train(args.compression, args.csv, args.output_dir)