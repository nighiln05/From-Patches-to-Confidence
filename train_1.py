import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from astra_attn_patch_dataset import ASTRA_AttnPatchRGBDataset, ALL_MACHINE_TYPES
from patch_attn_model import PatchAttentionCLModel, NTXentLoss

# ─── CONFIG ───────────────────────────────────────────────────────
ROOT_DIR = r"C:\Users\Nighil Natarajan\ckmam_proj\dcase"
CHECKPOINT_DIR  = "checkpoints_april10"

BATCH_SIZE      = 96
EPOCHS          = 170
LEARNING_RATE   = 2e-4
TEMPERATURE     = 0.05

MAX_PATCHES     = 32
STRIDE          = 16
EARLYSTOP_PAT   = 25
LR_PATIENCE     = 10
LR_FACTOR       = 0.5
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_ATTR = set(ALL_MACHINE_TYPES)

CONF_RATIO      = 0.85
EMA_MOMENTUM    = 0.9
HARD_WEIGHT     = 1.2
SOFT_WEIGHT     = 0.9

WARMUP_EPOCHS   = 8

BETA_BOUNDARY   = 0.02
BETA_CENTER     = 0.05
# ─────────────────────────────────────────────────────────────────


# ---------- DOMAIN GENERALIZATION ----------
def compute_cov(feat):
    feat = feat - feat.mean(dim=0, keepdim=True)
    cov = (feat.T @ feat) / (feat.size(0) - 1 + 1e-6)
    return cov

def coral_loss(source, target):
    cov_s = compute_cov(source)
    cov_t = compute_cov(target)
    return torch.mean((cov_s - cov_t) ** 2)

def mmd_loss(x, y):
    # 🔥 FIX: match sizes
    if x.size(0) != y.size(0):
        n = min(x.size(0), y.size(0))
        x = x[torch.randperm(x.size(0))[:n]]
        y = y[torch.randperm(y.size(0))[:n]]

    xx = torch.mm(x, x.T)
    yy = torch.mm(y, y.T)
    xy = torch.mm(x, y.T)

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    K = torch.exp(-0.5 * (rx.T + rx - 2 * xx))
    L = torch.exp(-0.5 * (ry.T + ry - 2 * yy))
    P = torch.exp(-0.5 * (rx.T + ry - 2 * xy))

    return K.mean() + L.mean() - 2 * P.mean()


def get_attr_dim(machine):
    if machine not in USE_ATTR:
        return 0
    import pandas as pd
    path = os.path.join(ROOT_DIR, machine, "attributes_00.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c not in {"filename", "file_name"}]
        return len(cols)
    return 0


def find_last_checkpoint():
    if not os.path.isdir(CHECKPOINT_DIR):
        return 0, None
    best_n, best_path = 0, None
    for fn in os.listdir(CHECKPOINT_DIR):
        if fn.startswith("epoch") and fn.endswith(".pth"):
            try:
                n = int(fn[len("epoch"):-4])
                if n > best_n:
                    best_n, best_path = n, os.path.join(CHECKPOINT_DIR, fn)
            except ValueError:
                continue
    return best_n, best_path


def compute_confidence_weights(z, ema_center=None, conf_ratio=0.85, hard_w=1.2, soft_w=0.9):
    with torch.no_grad():
        batch_center = z.mean(dim=0)

        if ema_center is None:
            center = batch_center
        else:
            center = EMA_MOMENTUM * ema_center + (1 - EMA_MOMENTUM) * batch_center

        dist = torch.norm(z - center.unsqueeze(0), dim=1)
        dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)

        k = int(conf_ratio * len(dist_norm))
        threshold = torch.topk(dist_norm, k=k, largest=False).values.max()

        confident_mask = dist_norm <= threshold
        uncertain_mask = ~confident_mask

        weights = torch.ones_like(dist_norm)
        weights[uncertain_mask] = soft_w

        hard_idx = torch.topk(dist_norm, k=len(dist_norm)//10, largest=True).indices
        weights[hard_idx] = hard_w

    return weights.detach(), center.detach()


def main():
    attr_dims       = [get_attr_dim(m) for m in USE_ATTR]
    global_attr_dim = max(attr_dims) if attr_dims else 0

    datasets = []
    for m in ALL_MACHINE_TYPES:
        ds = ASTRA_AttnPatchRGBDataset(
            ROOT_DIR, m, split="train",
            patch_size=32, stride=STRIDE,
            max_patches=MAX_PATCHES,
            global_attr_dim=global_attr_dim
        )
        datasets.append(ds)

    joint_ds = ConcatDataset(datasets)

    loader = DataLoader(
        joint_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = PatchAttentionCLModel(embed_dim=128, attr_dim=global_attr_dim).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE
    )

    criterion = NTXentLoss(temperature=TEMPERATURE)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    last_epoch, ckpt_path = find_last_checkpoint()
    start_epoch = last_epoch + 1
    best_loss   = float("inf")

    if ckpt_path is not None:
        print(f"🔁 Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        best_loss = ckpt.get("avg_loss", best_loss)
    else:
        print("🚀 Starting training from scratch")

    ema_center = None

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"[Epoch {epoch}/{EPOCHS}]", ncols=80)

        for batch in pbar:
            p1 = batch["patches_1"].to(DEVICE)
            p2 = batch["patches_2"].to(DEVICE)
            attrs = batch["attrs"].to(DEVICE)
            domain_labels = batch["domain"].to(DEVICE)

            optimizer.zero_grad()

            z1 = model(p1, p1.size(0), p1.size(1), attrs=attrs)
            z2 = model(p2, p2.size(0), p2.size(1), attrs=attrs)

            loss_raw = criterion(z1, z2)

            # ✅ FIXED DOMAIN SPLIT
            z_src = z1[domain_labels == 0]
            z_tgt = z1[domain_labels == 1]

            if len(z_src) > 1 and len(z_tgt) > 1:
                loss_coral = coral_loss(z_src, z_tgt)
                loss_mmd   = mmd_loss(z_src, z_tgt)
                loss_dg    = 0.05 * loss_coral + 0.02 * loss_mmd
            else:
                loss_dg = 0.0

            # ✅ DOMAIN ADVERSARIAL LOSS
            domain_logits = model.forward_domain(z1, lambda_=0.3)
            loss_domain = F.cross_entropy(domain_logits, domain_labels)

            if epoch <= WARMUP_EPOCHS + 3:
                loss = loss_raw + loss_dg + 0.1 * loss_domain
            else:
                progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)

                conf_ratio = 0.95 - 0.15 * progress
                hard_w     = 1.0 + 0.25 * progress
                soft_w     = 1.0 - 0.10 * progress

                weights, ema_center = compute_confidence_weights(
                    z1.detach(),
                    ema_center,
                    conf_ratio=conf_ratio,
                    hard_w=hard_w,
                    soft_w=soft_w
                )

                weights = weights.to(DEVICE)

                # ✅ FIXED SCAN WEIGHTING
                loss_scan = (loss_raw * weights).mean()

                lam = torch.distributions.Beta(0.5, 0.5).sample().to(DEVICE)
                z_mix = lam * z1 + (1 - lam) * z2

                if ema_center is None:
                    ema_center = z1.mean(dim=0).detach()

                dist_mix = torch.norm(z_mix - ema_center.unsqueeze(0), dim=1)
                loss_boundary = (lam - 0.5) * dist_mix.mean()

                center_pull = torch.norm(z1 - ema_center.unsqueeze(0), dim=1).mean()

                loss = loss_scan \
                     + BETA_BOUNDARY * loss_boundary \
                     + BETA_CENTER * center_pull \
                     + loss_dg \
                     + 0.1 * loss_domain

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        ckpt_out = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "avg_loss":    avg_loss
        }

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch{epoch}.pth")
        torch.save(ckpt_out, ckpt_path)

        ckpts = sorted(
            [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("epoch")],
            key=lambda x: int(x[len("epoch"):-4])
        )
        if len(ckpts) > 5:
            for fn in ckpts[:-5]:
                os.remove(os.path.join(CHECKPOINT_DIR, fn))

    print("✅ Training complete.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()