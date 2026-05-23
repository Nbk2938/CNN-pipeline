"""
Neubecker_T_OpenProject.py

Lunar image denoising — three progressive U-Net configurations:

  Config A  Basic U-Net: residual encoder, Sigmoid output, single-scale FFT loss,
            simple flip/rotation augmentation.
  Config B  + Squeeze-and-Excitation bottleneck, residual-noise Tanh output,
            multi-scale (ortho) FFT loss, noise-level augmentation, gradient clipping.
  Final     + Banding augmentation (synthetic horizontal/vertical line defects)
            matching the structured noise seen in test samples 7 and 42.

Outputs produced by this script  (all images saved under results/)
  config_comparison.png             training/val loss curves for the three configs
  kfold_curves.png                  5-fold cross-validation on the final model
  train_sample_{a,b,final}_{30,42}.png  per-config clean/noisy/denoised panels
  test_sample_{a,b,final}_{7,42}.png    per-config noisy/denoised panels
  cross_model_train_{30,42}.png     all three configs side-by-side (training samples)
  cross_model_test_{7,42}.png       all three configs side-by-side (test samples)
  residual_train_{30,42}.png        (clean − denoised) heatmaps per config
  metric_comparison.png             bar chart of validation MSE per config
  filters_{a,b,final}.png           first 16 learned conv filters per config
  prediction.npz                    denoised test images (float32, values in [0, 255])

Checkpoint / history caching
  Each config saves its training history as a .npy file alongside its checkpoint.
  If both files exist the training step is skipped so subsequent runs are fast.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters and paths
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
LR           = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS_A     = 150   # Config A training epochs (matches MLP_try1.py __main__)
EPOCHS_B     = 160   # Config B training epochs (matches MLP_try2.py __main__)
EPOCHS_FINAL = 200   # Final model — same run used for comparison plot and inference
KFOLD_K      = 5
KFOLD_EPOCHS = 100    # epochs per fold — FFT loss stabilises by ~epoch 20, 60 shows clear convergence
SEED         = 42

RESULTS_DIR = "results"

DATA_NOISY_TRAIN = "data/noisy_train_19k_harder.npy"
DATA_CLEAN_TRAIN = "data/clean_train_19k_harder.npy"
DATA_NOISY_TEST  = "data/noisy_val_1k_harder.npy"

# One checkpoint + history file per config — the final model checkpoint is also
# the one loaded for image visualisation and test-set inference.
CKPT_A     = "checkpoints/config_a_best.pt"
CKPT_B     = "checkpoints/config_b_best.pt"
CKPT_FINAL = "checkpoints/best_model.pt"

HIST_A     = "checkpoints/history_a.npy"
HIST_B     = "checkpoints/history_b.npy"
HIST_FINAL = "checkpoints/history_final.npy"
KFOLD_HIST = "checkpoints/kfold_histories.npy"

# ─────────────────────────────────────────────────────────────────────────────
# Shared convolutional building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive (Conv → BN → ReLU) layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDoubleConv(nn.Module):
    """DoubleConv with an identity (or 1×1 projection) skip connection."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = DoubleConv(in_ch, out_ch)
        self.skip  = (nn.Identity() if in_ch == out_ch
                      else nn.Conv2d(in_ch, out_ch, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class DownBlock(nn.Module):
    """Encoder stage: residual double conv then 2×2 max-pool."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ResidualDoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return feat, self.pool(feat)


class UpBlock(nn.Module):
    """Decoder stage: transposed conv, skip concatenation, double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            x  = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).view(x.size(0), -1, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Config A – Basic U-Net
# Architecture: 4-level residual encoder, plain bottleneck, Sigmoid direct output.
# Loss:         0.6 × MSE + 0.2 × L1 + 0.2 × single-scale FFT magnitude MSE.
# ─────────────────────────────────────────────────────────────────────────────

class UNetDenoiserA(nn.Module):
    """Config A: basic U-Net with Sigmoid direct output (no SE, no residual noise)."""

    def __init__(self, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.enc1 = DownBlock(1, 32)
        self.enc2 = DownBlock(32, 64)
        self.enc3 = DownBlock(64, 128)
        self.enc4 = DownBlock(128, 256)

        self.bottleneck = DoubleConv(256, 512)
        self.dropout    = nn.Dropout2d(p=dropout_p)

        self.dec4 = UpBlock(512, 256, 256)
        self.dec3 = UpBlock(256, 128, 128)
        self.dec2 = UpBlock(128, 64, 64)
        self.dec1 = UpBlock(64, 32, 32)

        self.head = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        x = self.dropout(self.bottleneck(x))
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)


class DenoisingLossA(nn.Module):
    """Config A loss: 0.6 MSE + 0.2 L1 + 0.2 single-scale FFT magnitude MSE."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1  = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fft = self.mse(torch.abs(torch.fft.fft2(pred)),
                       torch.abs(torch.fft.fft2(target)))
        return 0.6 * self.mse(pred, target) + 0.2 * self.l1(pred, target) + 0.2 * fft


# ─────────────────────────────────────────────────────────────────────────────
# Config B & Final – U-Net with SE attention and residual noise prediction
# Architecture: same encoder/decoder but with SE bottleneck and Tanh head that
#               predicts the noise residual: output = clamp(input − noise, 0, 1).
# Loss:         0.6 × MSE + 0.2 × L1 + 0.2 × multi-scale (1×, 2×, 4×) FFT MSE
#               with ortho normalisation (prevents FFT magnitude blow-up).
# ─────────────────────────────────────────────────────────────────────────────

class UNetDenoiser(nn.Module):
    """Config B / Final: U-Net with SE bottleneck and residual-noise Tanh head."""

    def __init__(self, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.enc1 = DownBlock(1, 32)
        self.enc2 = DownBlock(32, 64)
        self.enc3 = DownBlock(64, 128)
        self.enc4 = DownBlock(128, 256)

        self.bottleneck = DoubleConv(256, 512)
        self.dropout    = nn.Dropout2d(p=dropout_p)
        self.se         = SEBlock(512)

        self.dec4 = UpBlock(512, 256, 256)
        self.dec3 = UpBlock(256, 128, 128)
        self.dec2 = UpBlock(128, 64, 64)
        self.dec1 = UpBlock(64, 32, 32)

        self.head = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        x = self.se(self.dropout(self.bottleneck(x)))
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return (inp - self.head(x)).clamp(0.0, 1.0)


class DenoisingLoss(nn.Module):
    """Config B / Final loss: 0.6 MSE + 0.2 L1 + 0.2 multi-scale FFT (ortho)."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1  = nn.L1Loss()

    def _fft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = pred.new_zeros(1).squeeze()
        for s in (1, 2, 4):
            p = pred   if s == 1 else F.avg_pool2d(pred, s)
            t = target if s == 1 else F.avg_pool2d(target, s)
            total = total + self.mse(
                torch.abs(torch.fft.fft2(p, norm="ortho")),
                torch.abs(torch.fft.fft2(t, norm="ortho")),
            )
        return total / 3

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (0.6 * self.mse(pred, target)
                + 0.2 * self.l1(pred, target)
                + 0.2 * self._fft_loss(pred, target))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  (unified – banding augmentation toggled per config)
# ─────────────────────────────────────────────────────────────────────────────

class DenoisingDataset(Dataset):
    """Paired noisy/clean dataset with spatial and photometric augmentation.

    banding=True  adds synthetic horizontal/vertical line defects that mimic the
    structured sensor noise visible in test samples 7 and 42.
    """

    def __init__(
        self,
        noisy: np.ndarray,
        clean: Optional[np.ndarray],
        augment: bool = False,
        banding: bool = False,
        noise_prob: float = 0.6,
        noise_alpha: Tuple[float, float] = (0.9, 1.3),
    ) -> None:
        self.noisy       = torch.from_numpy(noisy).float()
        self.clean       = None if clean is None else torch.from_numpy(clean).float()
        self.augment     = augment
        self.banding     = banding
        self.noise_prob  = noise_prob
        self.noise_alpha = noise_alpha
        self.psz         = noisy.shape[-1]

    def __len__(self) -> int:
        return self.noisy.shape[0]

    def _augment(
        self,
        noisy: torch.Tensor,
        clean: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        rng = random.Random(torch.randint(0, 2**31 - 1, (1,)).item())

        # Spatial: random horizontal / vertical flip + occasional 90° rotation
        if rng.random() < 0.5:
            noisy = torch.flip(noisy, [2])
            if clean is not None:
                clean = torch.flip(clean, [2])
        if rng.random() < 0.5:
            noisy = torch.flip(noisy, [1])
            if clean is not None:
                clean = torch.flip(clean, [1])
        if rng.random() < 0.3:
            k = rng.choice([1, 2, 3])
            noisy = torch.rot90(noisy, k, [1, 2])
            if clean is not None:
                clean = torch.rot90(clean, k, [1, 2])

        # Circular shift: randomises the absolute spatial position of grid
        # artifacts without introducing boundary discontinuities.
        sh = rng.randint(0, self.psz)
        sw = rng.randint(0, self.psz)
        noisy = torch.roll(noisy, (sh, sw), (1, 2))
        if clean is not None:
            clean = torch.roll(clean, (sh, sw), (1, 2))

        # Noise-level jitter: scale the noise component by α in the per-config range.
        if clean is not None and rng.random() < self.noise_prob:
            alpha = rng.uniform(*self.noise_alpha)
            noisy = (clean + alpha * (noisy - clean)).clamp(0, 1)

        # Banding augmentation (Final config only): synthetic line defects.
        if self.banding and rng.random() < 0.6:
            _, h, w = noisy.shape
            width = 1  # initialised here so the grid section can reference it
            for _ in range(rng.randint(10, 20)):
                is_dark  = rng.random() < 0.8
                is_horiz = rng.random() < 0.5
                width    = (1 if rng.random() < 0.75
                            else max(3, min(8, int(rng.expovariate(0.8)) + 3)))
                max_d    = (rng.randint(8, 12) / 100.0 if width == 1
                            else max(0.2 / width, 0.05))
                delta    = rng.uniform(max_d * 0.4, max_d)
                sign     = -1 if is_dark else 1
                if is_horiz:
                    pos = rng.randint(0, h - width)
                    noisy[:, pos:pos + width, :] += sign * delta
                else:
                    pos = rng.randint(0, w - width)
                    noisy[:, :, pos:pos + width] += sign * delta
            # Dense periodic grid (emulates the regular grid artifact)
            if rng.random() < 0.6 and width < 4:
                spacing = rng.randint(10, 25)
                delta   = rng.uniform(0.05, 0.10)
                for pos in range(0, h, spacing):
                    noisy[:, pos:pos + 1, :] -= delta
                for pos in range(0, w, spacing):
                    noisy[:, :, pos:pos + 1] -= delta
                noisy = noisy.clip(0, 1)

        return noisy.clip(0, 1), clean

    def __getitem__(self, idx: int):
        noisy = self.noisy[idx]
        clean = None if self.clean is None else self.clean[idx]
        if self.augment:
            noisy, clean = self._augment(noisy, clean)
        return noisy if clean is None else (noisy, clean)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    noisy_tr = np.load(DATA_NOISY_TRAIN, allow_pickle=True).astype(np.float32) / 255.0
    clean_tr = np.load(DATA_CLEAN_TRAIN, allow_pickle=True).astype(np.float32) / 255.0
    noisy_te = np.load(DATA_NOISY_TEST,  allow_pickle=True).astype(np.float32) / 255.0
    return (np.expand_dims(noisy_tr, 1),
            np.expand_dims(clean_tr, 1),
            np.expand_dims(noisy_te, 1))


def make_loaders(
    noisy_tr: np.ndarray,
    clean_tr: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    banding: bool = False,
    noise_prob: float = 0.6,
    noise_alpha: Tuple[float, float] = (0.9, 1.3),
) -> Tuple[DataLoader, DataLoader]:
    pin = get_device().type == "cuda"
    train_loader = DataLoader(
        DenoisingDataset(noisy_tr[train_idx], clean_tr[train_idx],
                         augment=True, banding=banding,
                         noise_prob=noise_prob, noise_alpha=noise_alpha),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=pin,
    )
    val_loader = DataLoader(
        DenoisingDataset(noisy_tr[val_idx], clean_tr[val_idx],
                         augment=False, banding=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: bool = True,
) -> float:
    model.train()
    total = 0.0
    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(noisy), clean)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * noisy.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        total += criterion(model(noisy), clean).item() * noisy.size(0)
    return total / len(loader.dataset)


def _train_loop(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    ckpt_path: str,
    clip_grad: bool = True,
    log_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Inner training loop; saves best-val checkpoint and returns full history.

    If log_dir is given, train/val loss and learning rate are written to a
    TensorBoard event file so training can be monitored with `tensorboard --logdir runs`.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val  = float("inf")
    history: Dict[str, List[float]] = {"train": [], "val": []}
    writer    = (SummaryWriter(log_dir=log_dir)
                 if (log_dir is not None and SummaryWriter is not None) else None)

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, clip_grad)
        vl = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train"].append(tr)
        history["val"].append(vl)
        if writer is not None:
            writer.add_scalars("loss", {"train": tr, "val": vl}, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt_path)
        print(f"  epoch {epoch:03d}/{epochs}  train={tr:.5f}  val={vl:.5f}")

    if writer is not None:
        writer.close()
    print(f"  → best val: {best_val:.5f}")
    return history


def load_or_train(
    model: nn.Module,
    criterion: nn.Module,
    noisy_tr: np.ndarray,
    clean_tr: np.ndarray,
    ckpt_path: str,
    hist_path: str,
    epochs: int,
    banding: bool = False,
    clip_grad: bool = True,
    noise_prob: float = 0.6,
    noise_alpha: Tuple[float, float] = (0.9, 1.3),
    log_dir: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Load from cache if both checkpoint and history exist; otherwise train."""
    device = get_device()
    model  = model.to(device)

    if os.path.exists(ckpt_path) and os.path.exists(hist_path):
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True))
        history = np.load(hist_path, allow_pickle=True).item()
        print(f"  Loaded from cache: {ckpt_path}")
        return model, history

    indices = np.arange(noisy_tr.shape[0])
    tr_idx, val_idx = train_test_split(indices, test_size=0.2,
                                       random_state=SEED, shuffle=True)
    tl, vl = make_loaders(noisy_tr, clean_tr, tr_idx, val_idx,
                           banding=banding,
                           noise_prob=noise_prob, noise_alpha=noise_alpha)

    history = _train_loop(
        model, criterion, tl, vl, device, epochs, ckpt_path, clip_grad,
        log_dir=log_dir)
    np.save(hist_path, history)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True))
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# K-fold cross-validation (final model architecture + banding augmentation)
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold(
    noisy_tr: np.ndarray,
    clean_tr: np.ndarray,
) -> Tuple[List[List[float]], List[List[float]]]:
    device = get_device()
    kf     = KFold(n_splits=KFOLD_K, shuffle=True, random_state=SEED)
    fold_train_hist: List[List[float]] = []
    fold_val_hist:   List[List[float]] = []

    print("\nK-fold best validation losses per fold:")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(noisy_tr), 1):
        print(f"\n── Fold {fold}/{KFOLD_K} ──")
        model     = UNetDenoiser(dropout_p=0.2).to(device)
        criterion = DenoisingLoss()
        tl, vl    = make_loaders(noisy_tr, clean_tr, tr_idx, val_idx,
                                 banding=True, noise_prob=0.6, noise_alpha=(0.9, 1.3))
        ckpt      = f"checkpoints/kfold_fold{fold}.pt"
        history   = _train_loop(model, criterion, tl, vl, device,
                                 KFOLD_EPOCHS, ckpt, clip_grad=True,
                                 log_dir=f"runs/kfold/fold{fold}")
        fold_train_hist.append(history["train"])
        fold_val_hist.append(history["val"])
        print(f"  Fold {fold} summary — "
              f"train={min(history['train']):.5f}  val={min(history['val']):.5f}")

    return fold_train_hist, fold_val_hist


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_config_comparison(
    hist_a: Dict[str, List[float]],
    hist_b: Dict[str, List[float]],
    hist_final: Dict[str, List[float]],
) -> None:
    configs = [
        ("Config A\n(basic U-Net, Sigmoid, single FFT)",         hist_a,     "tab:blue",   "tab:cyan"),
        ("Config B\n(+SE, residual Tanh, multi-scale FFT)",       hist_b,     "tab:orange", "tab:red"),
        ("Final\n(+banding aug, grad clipping)",                  hist_final, "tab:green",  "tab:olive"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, hist, tc, vc) in zip(axes, configs):
        ep = range(1, len(hist["train"]) + 1)
        ax.plot(ep, hist["train"], color=tc, label="Train")
        ax.plot(ep, hist["val"],   color=vc, label="Val", linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training & Validation Loss — Three Configurations", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "config_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/config_comparison.png")


def plot_kfold_results(
    fold_train: List[List[float]],
    fold_val:   List[List[float]],
) -> None:
    tr_arr  = np.array(fold_train, dtype=np.float32)
    val_arr = np.array(fold_val,   dtype=np.float32)
    epochs  = np.arange(1, tr_arr.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(tr_arr.shape[0]):
        ax.plot(epochs, tr_arr[i],  color="tab:blue",   alpha=0.3, linewidth=1,
                label="Train (fold)" if i == 0 else "")
        ax.plot(epochs, val_arr[i], color="tab:orange", alpha=0.3, linewidth=1,
                label="Val (fold)"   if i == 0 else "")
    ax.plot(epochs, tr_arr.mean(0),  color="tab:blue",   linewidth=2.5, label="Mean Train")
    ax.plot(epochs, val_arr.mean(0), color="tab:orange", linewidth=2.5, label="Mean Val")
    ax.set_title(f"{KFOLD_K}-Fold Cross-Validation Loss — Final Model ({KFOLD_EPOCHS} epochs/fold)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, 0.030)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "kfold_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/kfold_curves.png")

    # Print per-fold summary table
    print(f"\n{'Fold':>5}  {'Best Train':>10}  {'Best Val':>10}")
    for i, (tr, vl) in enumerate(zip(fold_train, fold_val), 1):
        print(f"{i:>5}  {min(tr):>10.5f}  {min(vl):>10.5f}")
    print(f"{'Mean':>5}  {tr_arr.min(1).mean():>10.5f}  {val_arr.min(1).mean():>10.5f}")


def _imshow(ax: plt.Axes, img: np.ndarray, title: str) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def plot_training_samples(
    model: nn.Module,
    noisy_tr: np.ndarray,
    clean_tr: np.ndarray,
    device: torch.device,
    indices: Tuple[int, ...] = (30, 42),
    prefix: str = "",
) -> None:
    model.eval()
    for idx in indices:
        noisy    = noisy_tr[idx, 0]
        clean    = clean_tr[idx, 0]
        denoised = _denoise_single(model, noisy, device)
        n_mse    = _mse(clean, noisy)
        d_mse    = _mse(clean, denoised)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        _imshow(axes[0], clean,    "Clean")
        _imshow(axes[1], noisy,    f"Noisy  MSE={n_mse:.4f}")
        _imshow(axes[2], denoised, f"Denoised  MSE={d_mse:.4f}")
        fig.suptitle(f"Training Sample #{idx}", fontsize=11)
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"train_sample_{prefix}{idx}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}  "
              f"(noisy MSE={n_mse:.4f} → denoised MSE={d_mse:.4f})")


def plot_test_samples(
    model: nn.Module,
    noisy_te: np.ndarray,
    device: torch.device,
    indices: Tuple[int, ...] = (7, 42),
    prefix: str = "",
) -> None:
    model.eval()
    for idx in indices:
        noisy    = noisy_te[idx, 0]
        denoised = _denoise_tta(model, noisy, device)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        _imshow(axes[0], noisy,    "Noisy")
        _imshow(axes[1], denoised, "Denoised (TTA)")
        fig.suptitle(f"Test Sample #{idx}", fontsize=11)
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"test_sample_{prefix}{idx}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Additional diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_model_comparison(
    model_a: nn.Module,
    model_b: nn.Module,
    model_final: nn.Module,
    noisy_tr: np.ndarray,
    clean_tr: np.ndarray,
    noisy_te: np.ndarray,
    device: torch.device,
    indices_train: Tuple[int, ...] = (30, 42),
    indices_test: Tuple[int, ...]  = (7, 42),
) -> None:
    """Side-by-side denoising comparison across all three configs."""
    models = [("Config A", model_a), ("Config B", model_b), ("Final", model_final)]

    for idx in indices_train:
        noisy = noisy_tr[idx, 0]
        clean = clean_tr[idx, 0]
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        _imshow(axes[0], clean, "Clean")
        _imshow(axes[1], noisy, f"Noisy\nMSE={_mse(clean, noisy):.4f}")
        for ax, (name, m) in zip(axes[2:], models):
            den = _denoise_single(m, noisy, device)
            _imshow(ax, den, f"{name}\nMSE={_mse(clean, den):.4f}")
        fig.suptitle(f"Cross-Model Comparison — Training Sample #{idx}", fontsize=11)
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"cross_model_train_{idx}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")

    for idx in indices_test:
        noisy = noisy_te[idx, 0]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        _imshow(axes[0], noisy, "Noisy")
        for ax, (name, m) in zip(axes[1:], models):
            den = _denoise_tta(m, noisy, device)
            _imshow(ax, den, f"{name} (TTA)")
        fig.suptitle(f"Cross-Model Comparison — Test Sample #{idx}", fontsize=11)
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"cross_model_test_{idx}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")


def plot_residual_maps(
    model_a: nn.Module,
    model_b: nn.Module,
    model_final: nn.Module,
    noisy_tr: np.ndarray,
    clean_tr: np.ndarray,
    device: torch.device,
    indices: Tuple[int, ...] = (30, 42),
) -> None:
    """3×3 grid: one row per config (Clean | Denoised | Residual heatmap).

    Residual = clean − denoised.  Red = model over-smoothed, blue = over-predicted.
    """
    models = [("Config A", model_a), ("Config B", model_b), ("Final", model_final)]
    for idx in indices:
        noisy = noisy_tr[idx, 0]
        clean = clean_tr[idx, 0]
        fig, axes = plt.subplots(3, 3, figsize=(12, 11))
        for row, (name, m) in enumerate(models):
            denoised = _denoise_single(m, noisy, device)
            residual = clean - denoised
            _imshow(axes[row, 0], clean,    f"{name}\nClean")
            _imshow(axes[row, 1], denoised, f"{name}\nDenoised  MSE={_mse(clean, denoised):.4f}")
            im = axes[row, 2].imshow(residual, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
            axes[row, 2].set_title(f"{name}\nResidual  max|err|={np.abs(residual).max():.4f}",
                                   fontsize=9)
            axes[row, 2].axis("off")
            plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
        fig.suptitle(f"Residual Analysis — Training Sample #{idx}", fontsize=12)
        fig.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"residual_train_{idx}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")


@torch.no_grad()
def _val_mse(model: nn.Module, noisy: np.ndarray, clean: np.ndarray,
             device: torch.device) -> float:
    model.eval()
    pin = device.type == "cuda"
    ds = DenoisingDataset(noisy, clean, augment=False)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    total, n = 0.0, 0
    for noisy_b, clean_b in dl:
        pred = model(noisy_b.to(device, non_blocking=True)).cpu()
        total += ((pred - clean_b) ** 2).mean(dim=(1, 2, 3)).sum().item()
        n += noisy_b.size(0)
    return total / n


def plot_metric_comparison(
    model_a: nn.Module,
    model_b: nn.Module,
    model_final: nn.Module,
    noisy_tr: np.ndarray,
    clean_tr: np.ndarray,
    device: torch.device,
) -> None:
    """Bar chart of mean MSE on the held-out validation split for each config."""
    indices = np.arange(noisy_tr.shape[0])
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED, shuffle=True)
    nv, cv = noisy_tr[val_idx], clean_tr[val_idx]

    print("  Computing val MSE for Config A …")
    mse_a = _val_mse(model_a, nv, cv, device)
    print("  Computing val MSE for Config B …")
    mse_b = _val_mse(model_b, nv, cv, device)
    print("  Computing val MSE for Final …")
    mse_f = _val_mse(model_final, nv, cv, device)

    labels = ["Config A\n(basic U-Net)", "Config B\n(+SE, residual Tanh)", "Final\n(+banding aug)"]
    values = [mse_a, mse_b, mse_f]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5e-6,
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean MSE (validation set)")
    ax.set_title("Per-Configuration Validation MSE")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(RESULTS_DIR, "metric_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}  (A={mse_a:.5f}  B={mse_b:.5f}  Final={mse_f:.5f})")


def plot_learned_filters(model: nn.Module, filename: str = "filters.png"):
    """
    Plot first 16 kernels from the first Conv2d layer as a 4x4 grid.
    Each filter is normalized to [0, 1] for display.
    """
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break

    if first_conv is None:
        raise ValueError("No Conv2d layer found in model.")

    weights = first_conv.weight.detach().cpu().numpy()  # (out_c, in_c, k, k)
    filters = weights[:16, 0, :, :]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < filters.shape[0]:
            filt = filters[i]
            f_min, f_max = filt.min(), filt.max()
            if f_max > f_min:
                filt = (filt - f_min) / (f_max - f_min)
            else:
                filt = np.zeros_like(filt)
            ax.imshow(filt, cmap="gray")
            ax.set_title(f"F{i + 1}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("First 16 Learned Filters")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _denoise_single(
    model: nn.Module, img2d: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(img2d).float().unsqueeze(0).unsqueeze(0).to(device)
    return model(x).squeeze().cpu().numpy()


@torch.no_grad()
def _denoise_tta(
    model: nn.Module, img2d: np.ndarray, device: torch.device
) -> np.ndarray:
    """4-flip test-time augmentation.

    Rotations are excluded: the grid artifact is axis-aligned, so rot90 variants
    produce inconsistent predictions that average into blob artefacts.
    """
    model.eval()
    preds = []
    for hf in (False, True):
        for vf in (False, True):
            img = np.fliplr(img2d) if hf else img2d.copy()
            if vf:
                img = np.flipud(img)
            pred = _denoise_single(model, np.ascontiguousarray(img), device)
            if vf:
                pred = np.flipud(pred)
            if hf:
                pred = np.fliplr(pred)
            preds.append(pred)
    return np.mean(preds, axis=0)


@torch.no_grad()
def denoise_test_set(
    model: nn.Module,
    noisy_te: np.ndarray,
    device: torch.device,
    out_path: str = "prediction.npz",
) -> None:
    model.eval()
    pin = device.type == "cuda"
    ds  = DenoisingDataset(noisy_te, clean=None, augment=False)
    dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=0, pin_memory=pin)
    preds = []
    for batch in dl:
        batch = batch.to(device, non_blocking=True)
        preds.append(model(batch).cpu())
    denoised = torch.cat(preds, 0).squeeze(1).numpy()
    denoised = np.clip(denoised, 0.0, 1.0)
    output   = (denoised * 255.0).astype(np.float32)
    np.savez(out_path, denoised_images=output)
    print(f"Saved {out_path}  shape={output.shape}  "
          f"range=[{output.min():.1f}, {output.max():.1f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = get_device()
    print(f"Device: {device}\n")

    noisy_tr, clean_tr, noisy_te = load_data()
    print(f"Train shape: {noisy_tr.shape}  Test shape: {noisy_te.shape}\n")

    # ── Config A: Basic U-Net ─────────────────────────────────────────────────
    print("=" * 60)
    print("Config A — Basic U-Net (Sigmoid, single-scale FFT loss)")
    print("=" * 60)
    model_a, hist_a = load_or_train(
        UNetDenoiserA(dropout_p=0.2),       # matches MLP_try1.py run_training
        DenoisingLossA(),
        noisy_tr, clean_tr,
        CKPT_A, HIST_A,
        epochs=EPOCHS_A,
        banding=False,
        clip_grad=False,                    # MLP_try1.py has no grad clipping
        noise_prob=0.5,                     # MLP_try1.py: rng.random() < 0.5
        noise_alpha=(0.7, 1.3),             # MLP_try1.py: rng.uniform(0.7, 1.3)
        log_dir="runs/config_a",
    )

    # ── Config B: + SE + residual Tanh + multi-scale FFT ─────────────────────
    print("\n" + "=" * 60)
    print("Config B — + SE attention, residual-noise Tanh, multi-scale FFT")
    print("=" * 60)
    model_b, hist_b = load_or_train(
        UNetDenoiser(dropout_p=0.2),
        DenoisingLoss(),
        noisy_tr, clean_tr,
        CKPT_B, HIST_B,
        epochs=EPOCHS_B,
        banding=False,                      # MLP_try2.py has banding commented out
        clip_grad=True,
        noise_prob=0.6,                     # MLP_try2.py: rng.random() < 0.6
        noise_alpha=(0.9, 1.5),             # MLP_try2.py: rng.uniform(0.9, 1.5)
        log_dir="runs/config_b",
    )

    # ── Final: + banding augmentation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final — Config B + banding augmentation")
    print("=" * 60)
    model_final, hist_final = load_or_train(
        UNetDenoiser(dropout_p=0.2),
        DenoisingLoss(),
        noisy_tr, clean_tr,
        CKPT_FINAL, HIST_FINAL,
        epochs=EPOCHS_FINAL,
        banding=True,
        clip_grad=True,
        noise_prob=0.6,                     # MLP_skelleton.py: rng.random() < 0.6
        noise_alpha=(0.9, 1.3),             # MLP_skelleton.py: rng.uniform(0.9, 1.3)
        log_dir="runs/final",
    )

    # ── Configuration comparison plot ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Plotting configuration comparison")
    print("=" * 60)
    plot_config_comparison(hist_a, hist_b, hist_final)

    # ── K-fold cross-validation ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{KFOLD_K}-Fold Cross-Validation — Final model ({KFOLD_EPOCHS} epochs/fold)")
    print("=" * 60)
    if os.path.exists(KFOLD_HIST):
        kfold_data  = np.load(KFOLD_HIST, allow_pickle=True).item()
        fold_train  = kfold_data["train"]
        fold_val    = kfold_data["val"]
        print("Loaded k-fold histories from cache.")
    else:
        fold_train, fold_val = run_kfold(noisy_tr, clean_tr)
        np.save(KFOLD_HIST, {"train": fold_train, "val": fold_val})
    plot_kfold_results(fold_train, fold_val)

    # ── Image visualisations (all three models) ──────────────────────────────
    print("\n" + "=" * 60)
    print("Image visualisations — Config A")
    print("=" * 60)
    plot_training_samples(model_a, noisy_tr, clean_tr, device,
                          indices=(30, 42), prefix="a_")
    plot_test_samples(model_a, noisy_te, device, indices=(7, 42), prefix="a_")

    print("\n" + "=" * 60)
    print("Image visualisations — Config B")
    print("=" * 60)
    plot_training_samples(model_b, noisy_tr, clean_tr, device,
                          indices=(30, 42), prefix="b_")
    plot_test_samples(model_b, noisy_te, device, indices=(7, 42), prefix="b_")

    print("\n" + "=" * 60)
    print("Image visualisations — Final")
    print("=" * 60)
    plot_training_samples(model_final, noisy_tr, clean_tr, device,
                          indices=(30, 42), prefix="final_")
    plot_test_samples(model_final, noisy_te, device, indices=(7, 42), prefix="final_")

    # ── Cross-model comparison panels ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Cross-model comparison (all configs side by side)")
    print("=" * 60)
    plot_cross_model_comparison(model_a, model_b, model_final,
                                noisy_tr, clean_tr, noisy_te, device,
                                indices_train=(30, 42), indices_test=(7, 42))

    # ── Residual maps (clean − denoised) for training samples ────────────────
    print("\n" + "=" * 60)
    print("Residual maps — training samples")
    print("=" * 60)
    plot_residual_maps(model_a, model_b, model_final,
                       noisy_tr, clean_tr, device, indices=(30, 42))

    # ── Per-config validation MSE bar chart ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Metric comparison (val MSE)")
    print("=" * 60)
    plot_metric_comparison(model_a, model_b, model_final,
                           noisy_tr, clean_tr, device)

    # ── Learned filters ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Learned filters")
    print("=" * 60)
    for tag, m in [("a", model_a), ("b", model_b), ("final", model_final)]:
        plot_learned_filters(m, filename=os.path.join(RESULTS_DIR, f"filters_{tag}.png"))
        print(f"Saved {RESULTS_DIR}/filters_{tag}.png")

    # ── Test-set denoising → prediction.npz ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Denoising test set")
    print("=" * 60)
    denoise_test_set(model_final, noisy_te, device, out_path="prediction.npz")

    print("\n✓ All done.")
