import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils.visualization import plot_image_comparison, plot_learned_filters


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) x2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDoubleConv(nn.Module):
    """DoubleConv with a residual skip (1x1 projection when channels differ)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = DoubleConv(in_channels, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class DownBlock(nn.Module):
    """Encoder: residual double conv + max-pool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ResidualDoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        return features, self.pool(features)


class UpBlock(nn.Module):
    """Decoder: transposed conv + skip concat + double conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
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


class UNetDenoiser(nn.Module):
    """U-Net with residual encoder for grayscale denoising. I/O: (N,1,H,W) in [0,1]."""

    def __init__(self, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.enc1 = DownBlock(1, 32)
        self.enc2 = DownBlock(32, 64)
        self.enc3 = DownBlock(64, 128)
        self.enc4 = DownBlock(128, 256)

        self.bottleneck = DoubleConv(256, 512)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.se = SEBlock(512)

        self.dec4 = UpBlock(512, 256, 256)
        self.dec3 = UpBlock(256, 128, 128)
        self.dec2 = UpBlock(128, 64, 64)
        self.dec1 = UpBlock(64, 32, 32)

        self.head = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Tanh())

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


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DenoisingLoss(nn.Module):
    """0.6*MSE + 0.2*L1 + 0.2*multi-scale FFT-magnitude MSE.

    Multi-scale FFT (full, /2, /4) targets both fine and coarse grid periodicity.
    """

    def __init__(self, mse_w: float = 0.6, l1_w: float = 0.2, fft_w: float = 0.3) -> None:
        super().__init__()
        self.mse_w, self.l1_w, self.fft_w = mse_w, l1_w, fft_w
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def _fft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = pred.new_zeros(1).squeeze()
        for stride in (1, 2, 4):
            p = pred if stride == 1 else F.avg_pool2d(pred, stride)
            t = target if stride == 1 else F.avg_pool2d(target, stride)
            #total = total + self.mse(torch.abs(torch.fft.fft2(p)), torch.abs(torch.fft.fft2(t)))
            total = total + self.mse(torch.abs(torch.fft.fft2(p, norm="ortho")), torch.abs(torch.fft.fft2(t, norm="ortho")))
        return total / 3

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.mse_w * self.mse(pred, target)
            + self.l1_w * self.l1(pred, target)
            + self.fft_w * self._fft_loss(pred, target)
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DenoisingDataset(Dataset):
    """Paired noisy/clean dataset with flips, rotations, and scale jitter."""

    def __init__(
        self, noisy: np.ndarray, clean: Optional[np.ndarray], augment: bool = False
    ) -> None:
        self.noisy = torch.from_numpy(noisy).float()
        self.clean = None if clean is None else torch.from_numpy(clean).float()
        self.augment = augment
        self.patch_size = noisy.shape[-1]

    def __len__(self) -> int:
        return self.noisy.shape[0]

    def _augment(
        self, noisy: torch.Tensor, clean: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        rng = random.Random(torch.randint(0, 2**31 - 1, (1,)).item())

        if rng.random() < 0.5:
            noisy = torch.flip(noisy, dims=[2])
            if clean is not None:
                clean = torch.flip(clean, dims=[2])

        if rng.random() < 0.5:
            noisy = torch.flip(noisy, dims=[1])
            if clean is not None:
                clean = torch.flip(clean, dims=[1])

        if rng.random() < 0.3:
            k = rng.choice([1, 2, 3])
            noisy = torch.rot90(noisy, k=k, dims=[1, 2])
            if clean is not None:
                clean = torch.rot90(clean, k=k, dims=[1, 2])

        # Circular shift: roll both noisy and clean together so pixel correspondence
        # is preserved. Randomises absolute position of grid artifacts spatially.
        shift_h = rng.randint(0, self.patch_size)
        shift_w = rng.randint(0, self.patch_size)
        noisy = torch.roll(noisy, shifts=(shift_h, shift_w), dims=(1, 2))
        if clean is not None:
            clean = torch.roll(clean, shifts=(shift_h, shift_w), dims=(1, 2))

        # Noise level augmentation: ±30% scale.
        if clean is not None and rng.random() < 0.6:
            alpha = rng.uniform(0.9, 1.5)
            noisy = (clean + alpha * (noisy - clean)).clamp(0, 1)

        # Banding augmentation: random bands + dense regular grid to simulate
        # sensor line defects (e.g. test samples 7, 42).
        """if rng.random() < 0.6:
            _, h, w = noisy.shape
            for _ in range(rng.randint(30, 50)):
                is_dark  = rng.random() < 0.8
                is_horiz = rng.random() < 0.5
                width    = max(1, min(10, int(rng.expovariate(1.5))))
                max_d    = rng.randint(8,12)/100.0 if width==1 else min(0.35 / width ** 1.5, 0.12)
                delta    = rng.uniform(max_d * 0.4, max_d)
                sign     = -1 if is_dark else 1
                if is_horiz:
                    pos = rng.randint(0, h - width)
                    noisy[:, pos:pos + width, :] += sign * delta
                else:
                    pos = rng.randint(0, w - width)
                    noisy[:, :, pos:pos + width] += sign * delta
            if rng.random() < 0.6:
                spacing = rng.randint(10, 25)
                delta   = rng.uniform(0.05, 0.10)
                for pos in range(0, h, spacing):
                    noisy[:, pos:pos + 1, :] -= delta
                for pos in range(0, w, spacing):
                    noisy[:, :, pos:pos + 1] -= delta
                noisy = noisy.clip(0, 1)"""

        return noisy.clip(0, 1), clean

    def __getitem__(self, idx: int):
        noisy = self.noisy[idx]
        clean = None if self.clean is None else self.clean[idx]
        if self.augment:
            noisy, clean = self._augment(noisy, clean)
        return noisy if clean is None else (noisy, clean)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess_data(
    train_noisy_path: str = "data/noisy_train_19k_harder.npy",
    train_clean_path: str = "data/clean_train_19k_harder.npy",
    test_noisy_path: str = "data/noisy_val_1k_harder.npy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, normalise, and add channel dim. Train stays at native 128x128."""
    noisy_train = np.load(train_noisy_path, allow_pickle=True).astype(np.float32) / 255.0
    clean_train = np.load(train_clean_path, allow_pickle=True).astype(np.float32) / 255.0
    noisy_test  = np.load(test_noisy_path,  allow_pickle=True).astype(np.float32) / 255.0

    noisy_train = np.expand_dims(noisy_train, axis=1)
    clean_train = np.expand_dims(clean_train, axis=1)
    noisy_test  = np.expand_dims(noisy_test,  axis=1)

    return noisy_train, clean_train, noisy_test


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(noisy), clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        total += criterion(model(noisy), clean).item() * noisy.size(0)
    return total / len(loader.dataset)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(
    noisy_train: np.ndarray,
    clean_train: np.ndarray,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    num_epochs: int = 150,
    save_dir: str = "checkpoints",
    log_dir: str = "runs/training",
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))

    indices = np.arange(noisy_train.shape[0])
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed, shuffle=True)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        DenoisingDataset(noisy_train[train_idx], clean_train[train_idx], augment=True),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        DenoisingDataset(noisy_train[val_idx], clean_train[val_idx], augment=False),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory,
    )

    model     = UNetDenoiser(dropout_p=0.2).to(device)
    criterion = DenoisingLoss(mse_w=0.6, l1_w=0.2, fft_w=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    writer         = SummaryWriter(log_dir=log_dir)
    checkpoint     = os.path.join(save_dir, "best_model.pt")
    best_val       = float("inf")
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch:03d}/{num_epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint)

    writer.close()
    print(f"\nBest val_loss: {best_val:.6f}  —  saved to {checkpoint}")
    return model, history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_test_loader(noisy_test: np.ndarray, batch_size: int = 32) -> DataLoader:
    return DataLoader(
        DenoisingDataset(noisy_test, clean=None, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )


@torch.no_grad()
def denoise_single_image(
    model: nn.Module, noisy_2d: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(noisy_2d).float().unsqueeze(0).unsqueeze(0).to(device)
    return model(x).squeeze().cpu().numpy()


@torch.no_grad()
def denoise_with_patches(
    model: nn.Module,
    noisy_2d: np.ndarray,
    device: torch.device,
    patch_size: int = 128,
    overlap: int = 32,
) -> np.ndarray:
    """Overlapping-patch inference — averages predictions in overlapping regions."""
    model.eval()
    h, w   = noisy_2d.shape
    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    stride = patch_size - overlap

    ys = list(range(0, h - patch_size + 1, stride))
    xs = list(range(0, w - patch_size + 1, stride))
    if not ys or ys[-1] + patch_size < h:
        ys.append(max(0, h - patch_size))
    if not xs or xs[-1] + patch_size < w:
        xs.append(max(0, w - patch_size))

    for y in ys:
        for x in xs:
            patch = noisy_2d[y : y + patch_size, x : x + patch_size]
            inp   = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
            pred  = model(inp).squeeze().cpu().numpy()
            output[y : y + patch_size, x : x + patch_size] += pred
            weight[y : y + patch_size, x : x + patch_size] += 1.0

    return output / np.maximum(weight, 1.0)


@torch.no_grad()
def denoise_tta(
    model: nn.Module, noisy_2d: np.ndarray, device: torch.device
) -> np.ndarray:
    """Test-time augmentation: average over 4 flip variants only.

    Rotations are intentionally excluded — the grid artifact is axis-aligned,
    so rot90 produces inconsistent predictions that average into blob artifacts.
    """
    model.eval()
    preds = []
    for hflip in (False, True):
        for vflip in (False, True):
            img = noisy_2d
            if hflip:
                img = np.fliplr(img)
            if vflip:
                img = np.flipud(img)
            pred = denoise_single_image(model, np.ascontiguousarray(img), device)
            if vflip:
                pred = np.flipud(pred)
            if hflip:
                pred = np.fliplr(pred)
            preds.append(pred)
    return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    noisy_train, clean_train, noisy_test = load_and_preprocess_data(
        train_noisy_path="data/noisy_train_19k_harder.npy",
        train_clean_path="data/clean_train_19k_harder.npy",
        test_noisy_path="data/noisy_val_1k_harder.npy",
    )
    print("Train noisy:", noisy_train.shape)
    print("Train clean:", clean_train.shape)
    print("Test noisy: ", noisy_test.shape)

    model, history = run_training(
        noisy_train=noisy_train,
        clean_train=clean_train,
        batch_size=32,
        lr=5e-4,
        num_epochs=160,
        save_dir="checkpoints",
        log_dir="runs/training",
    )

    device = get_device()
    checkpoint = "checkpoints/best_model.pt"
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded best weights from {checkpoint}")

    aug_ds = DenoisingDataset(noisy_train, clean_train, augment=True)
    for idx in [0, 5, 7, 13, 30, 42, 77]:
        noisy_img   = noisy_train[idx, 0]
        clean_img   = clean_train[idx, 0]
        denoised    = denoise_single_image(model, noisy_img, device)
        aug_noisy, aug_clean = aug_ds[idx]
        plot_image_comparison(
            noisy=noisy_img, clean=clean_img, denoised=denoised,
            title="Training Sample", filename=f"training_sample_{idx}.png",
            augmented=aug_noisy[0].numpy(),
            aug_clean=aug_clean[0].numpy(),
        )

    for idx in [0, 5, 7, 13, 30, 42, 77]:
        noisy_img = noisy_test[idx, 0]
        denoised  = denoise_tta(model, noisy_img, device)
        plot_image_comparison(
            noisy=noisy_img, clean=None, denoised=denoised,
            title="Test Sample", filename=f"test_sample_{idx}.png",
        )

    plot_learned_filters(model, filename="filters.png")
