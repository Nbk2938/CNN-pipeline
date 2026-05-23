import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    from pytorch_msssim import ssim
except ImportError as exc:
    raise ImportError(
        "This script requires pytorch-msssim. Install it with: pip install pytorch-msssim"
    ) from exc


class ResidualDoubleConv(nn.Module):
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
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ResidualDoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        downsampled = self.pool(features)
        return features, downsampled


class DoubleConv(nn.Module):
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


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            diff_h = skip.size(-2) - x.size(-2)
            diff_w = skip.size(-1) - x.size(-1)
            x = nn.functional.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ConfigCUNet(nn.Module):
    """Deep U-Net with residual encoder blocks and bottleneck dropout."""

    def __init__(self, dropout_p: float = 0.1) -> None:
        super().__init__()

        self.enc1 = DownBlock(1, 32)
        self.enc2 = DownBlock(32, 64)
        self.enc3 = DownBlock(64, 128)
        self.enc4 = DownBlock(128, 256)

        self.bottleneck = DoubleConv(256, 512)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.dec4 = UpBlock(512, 256, 256)
        self.dec3 = UpBlock(256, 128, 128)
        self.dec2 = UpBlock(128, 64, 64)
        self.dec1 = UpBlock(64, 32, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.final_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)
        x = self.dropout(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        x = self.final_conv(x)
        return self.final_act(x)


class ConfigCLoss(nn.Module):
    """0.6*MSE + 0.2*L1 + 0.2*(-SSIM)."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_term = self.mse(prediction, target)
        l1_term = self.l1(prediction, target)
        ssim_term = -ssim(prediction, target, data_range=1.0, size_average=True)
        return 0.6 * mse_term + 0.2 * l1_term + 0.2 * ssim_term


class DenoisingDataset(Dataset):
    def __init__(
        self,
        noisy: np.ndarray,
        clean: Optional[np.ndarray],
        augment: bool = False,
    ) -> None:
        self.noisy = torch.from_numpy(noisy).float()
        self.clean = None if clean is None else torch.from_numpy(clean).float()
        self.augment = augment

    def __len__(self) -> int:
        return self.noisy.shape[0]

    def _augment_pair(
        self, noisy_img: torch.Tensor, clean_img: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item()
        rng = random.Random(seed)

        if rng.random() < 0.5:
            noisy_img = torch.flip(noisy_img, dims=[2])
            if clean_img is not None:
                clean_img = torch.flip(clean_img, dims=[2])

        if rng.random() < 0.5:
            noisy_img = torch.flip(noisy_img, dims=[1])
            if clean_img is not None:
                clean_img = torch.flip(clean_img, dims=[1])

        if rng.random() < 0.3:
            k = rng.choice([1, 2, 3])
            noisy_img = torch.rot90(noisy_img, k=k, dims=[1, 2])
            if clean_img is not None:
                clean_img = torch.rot90(clean_img, k=k, dims=[1, 2])

        return noisy_img, clean_img

    def __getitem__(self, idx: int):
        noisy_img = self.noisy[idx]
        clean_img = None if self.clean is None else self.clean[idx]

        if self.augment:
            noisy_img, clean_img = self._augment_pair(noisy_img, clean_img)

        if clean_img is None:
            return noisy_img
        return noisy_img, clean_img


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_train_data(
    noisy_path: str = "data/noisy_train_19k_harder.npy",
    clean_path: str = "data/clean_train_19k_harder.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    noisy = np.load(noisy_path).astype(np.float32) / 255.0
    clean = np.load(clean_path).astype(np.float32) / 255.0

    noisy = np.expand_dims(noisy, axis=1)
    clean = np.expand_dims(clean, axis=1)
    return noisy, clean


def load_test_data(test_path: str = "data/noisy_val_1k_harder.npy") -> np.ndarray:
    noisy = np.load(test_path).astype(np.float32) / 255.0
    noisy = np.expand_dims(noisy, axis=1)
    return noisy


def evaluate_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    mse_fn = nn.MSELoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for noisy_batch, clean_batch in loader:
            noisy_batch = noisy_batch.to(device, non_blocking=True)
            clean_batch = clean_batch.to(device, non_blocking=True)
            pred = model(noisy_batch)
            total_loss += mse_fn(pred, clean_batch).item()
            n_samples += noisy_batch.size(0)

    return total_loss / n_samples


def train_config_c_full(
    noisy: np.ndarray,
    clean: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    save_path: str = "best_config_c.pt",
    seed: int = 42,
) -> Tuple[nn.Module, float, float]:
    device = get_device()
    indices = np.arange(noisy.shape[0])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    train_dataset = DenoisingDataset(noisy[train_idx], clean[train_idx], augment=True)
    val_dataset = DenoisingDataset(noisy[val_idx], clean[val_idx], augment=False)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = ConfigCUNet(dropout_p=0.1).to(device)
    criterion = ConfigCLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(device, non_blocking=True)
            clean_batch = clean_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(noisy_batch)
            loss = criterion(pred, clean_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_batch, clean_batch in val_loader:
                noisy_batch = noisy_batch.to(device, non_blocking=True)
                clean_batch = clean_batch.to(device, non_blocking=True)
                pred = model(noisy_batch)
                loss = criterion(pred, clean_batch)
                val_loss += loss.item() * noisy_batch.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))

    final_train_mse = evaluate_mse(model, train_loader, device)
    final_val_mse = evaluate_mse(model, val_loader, device)

    return model, final_train_mse, final_val_mse


def load_best_model_from_kfold(
    model: nn.Module,
    val_losses_path: str = "kfold_best_val_losses.npy",
    checkpoint_dir: str = "checkpoints",
) -> Tuple[nn.Module, int, float]:
    """
    Loads the best k-fold model by selecting the fold with the lowest validation loss.

    Expects val_losses_path to contain a NumPy array of shape (5,) where each value is
    the best val loss for that fold, and checkpoints named best_unet_fold{fold}.pt.
    """
    if not os.path.exists(val_losses_path):
        raise FileNotFoundError(
            f"Missing {val_losses_path}. Save fold losses there to select best fold automatically."
        )

    fold_losses = np.load(val_losses_path).astype(np.float32)
    if fold_losses.shape[0] != 5:
        raise ValueError("kfold_best_val_losses.npy must contain exactly 5 fold losses.")

    best_fold = int(np.argmin(fold_losses)) + 1
    best_loss = float(fold_losses[best_fold - 1])
    ckpt_path = os.path.join(checkpoint_dir, f"best_unet_fold{best_fold}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Expected checkpoint not found: {ckpt_path}")

    device = get_device()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, best_fold, best_loss


def run_inference_and_save(
    model: nn.Module,
    test_noisy: np.ndarray,
    batch_size: int = 16,
    out_path: str = "prediction.npz",
) -> np.ndarray:
    device = get_device()
    model = model.to(device)
    model.eval()

    test_dataset = DenoisingDataset(test_noisy, clean=None, augment=False)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    outputs = []
    with torch.no_grad():
        for noisy_batch in loader:
            noisy_batch = noisy_batch.to(device, non_blocking=True)
            pred = model(noisy_batch)
            outputs.append(pred.cpu())

    denoised = torch.cat(outputs, dim=0).numpy()  # (N,1,H,W)
    denoised = denoised.squeeze(1)  # (N,H,W)
    denoised = np.clip(denoised, 0.0, 1.0)

    output_array = (denoised * 255.0).astype(np.float32)
    np.savez(out_path, denoised_images=output_array)

    return output_array


def main() -> None:
    # End-to-end pipeline requested: load data -> train Config C (100 epochs) -> infer -> save npz
    noisy_train, clean_train = load_train_data(
        noisy_path="data/noisy_train_19k_harder.npy",
        clean_path="data/clean_train_19k_harder.npy",
    )
    noisy_test = load_test_data(test_path="data/noisy_val_1k_harder.npy")

    print("Train noisy shape:", noisy_train.shape)
    print("Train clean shape:", clean_train.shape)
    print("Test noisy shape:", noisy_test.shape)

    model, final_train_mse, final_val_mse = train_config_c_full(
        noisy=noisy_train,
        clean=clean_train,
        epochs=100,
        batch_size=16,
        save_path="best_config_c.pt",
        seed=42,
    )

    output_array = run_inference_and_save(
        model=model,
        test_noisy=noisy_test,
        batch_size=16,
        out_path="prediction.npz",
    )

    print("Saved prediction file: prediction.npz")
    print("Output shape:", output_array.shape)
    print("Output dtype:", output_array.dtype)
    print("Output min/max:", float(output_array.min()), float(output_array.max()))

    print(f"Final train MSE: {final_train_mse:.6f}")
    print(f"Final val MSE: {final_val_mse:.6f}")


if __name__ == "__main__":
    main()
