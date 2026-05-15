import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from visualization import plot_image_comparison, plot_kfold_results, plot_learned_filters

try:
    from pytorch_msssim import ssim
except ImportError:
    ssim = None


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) x2 with padding=1."""

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
    """Two conv blocks with a residual skip projection when channel count differs."""

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
    """Encoder block: double conv, then max-pool for downsampling."""

    def __init__(
        self, in_channels: int, out_channels: int, residual: bool = False, use_batchnorm: bool = True
    ) -> None:
        super().__init__()
        if residual:
            self.conv = ResidualDoubleConv(in_channels, out_channels)
        elif use_batchnorm:
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        downsampled = self.pool(features)
        return features, downsampled


class UpBlock(nn.Module):
    """Decoder block: transposed conv upsample, skip concat, then double conv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        if use_batchnorm:
            self.conv = DoubleConv(out_channels + skip_channels, out_channels)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle possible odd/even spatial mismatches safely.
        if x.shape[-2:] != skip.shape[-2:]:
            diff_h = skip.size(-2) - x.size(-2)
            diff_w = skip.size(-1) - x.size(-1)
            x = nn.functional.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetDenoiser(nn.Module):
    """
    U-Net for grayscale image denoising.

    Input shape: (N, 1, H, W), with H=W=64 or 128 (or any size divisible by 16).
    Output shape: (N, 1, H, W), values in [0, 1].
    """

    def __init__(
        self,
        dropout_p: float = 0.0,
        use_batchnorm: bool = True,
        residual_encoder: bool = False,
    ) -> None:
        super().__init__()

        # Encoder: 1 -> 32 -> 64 -> 128 -> 256
        self.enc1 = DownBlock(1, 32, residual=residual_encoder, use_batchnorm=use_batchnorm)
        self.enc2 = DownBlock(32, 64, residual=residual_encoder, use_batchnorm=use_batchnorm)
        self.enc3 = DownBlock(64, 128, residual=residual_encoder, use_batchnorm=use_batchnorm)
        self.enc4 = DownBlock(128, 256, residual=residual_encoder, use_batchnorm=use_batchnorm)

        # Bottleneck: 512 channels + dropout only here.
        if use_batchnorm:
            self.bottleneck = DoubleConv(256, 512)
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.bottleneck_dropout = nn.Dropout2d(p=dropout_p)

        # Decoder: upsample + skip concat + double conv
        self.dec4 = UpBlock(512, 256, 256, use_batchnorm=use_batchnorm)
        self.dec3 = UpBlock(256, 128, 128, use_batchnorm=use_batchnorm)
        self.dec2 = UpBlock(128, 64, 64, use_batchnorm=use_batchnorm)
        self.dec1 = UpBlock(64, 32, 32, use_batchnorm=use_batchnorm)

        # Final 1x1 conv + sigmoid
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)
        x = self.bottleneck_dropout(x)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        x = self.final_conv(x)
        return self.final_activation(x)


class CombinedLoss(nn.Module):
    """0.8 * MSELoss + 0.2 * L1Loss."""

    def __init__(self, mse_weight: float = 0.8, l1_weight: float = 0.2) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse_weight * self.mse(prediction, target) + self.l1_weight * self.l1(
            prediction, target
        )


class MSEPlusL1PlusSSIMLoss(nn.Module):
    """0.6*MSE + 0.2*L1 + 0.2*(-SSIM)."""

    def __init__(self, mse_weight: float = 0.6, l1_weight: float = 0.2, ssim_weight: float = 0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if ssim is None:
            raise ImportError(
                "pytorch_msssim is required for Config C. Install with: pip install pytorch-msssim"
            )
        ssim_term = -ssim(prediction, target, data_range=1.0, size_average=True)
        return (
            self.mse_weight * self.mse(prediction, target)
            + self.l1_weight * self.l1(prediction, target)
            + self.ssim_weight * ssim_term
        )


class DenoisingDataset(Dataset):
    """Paired noisy/clean dataset with synchronized augmentations."""

    def __init__(
        self, noisy: np.ndarray, clean: Optional[np.ndarray], augment: bool = False
    ) -> None:
        self.noisy = torch.from_numpy(noisy).float()
        self.clean = None if clean is None else torch.from_numpy(clean).float()
        self.augment = augment

    def __len__(self) -> int:
        return self.noisy.shape[0]

    def _apply_augment_pair(
        self, noisy_img: torch.Tensor, clean_img: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # One seed per sample ensures noisy/clean transforms are identical.
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
            noisy_img, clean_img = self._apply_augment_pair(noisy_img, clean_img)

        if clean_img is None:
            return noisy_img
        return noisy_img, clean_img


def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_and_loss() -> tuple[UNetDenoiser, CombinedLoss, torch.device]:
    """Convenience helper to construct a device-placed model and loss function."""
    device = get_device()
    model = UNetDenoiser(dropout_p=0.1, use_batchnorm=True, residual_encoder=False).to(device)
    criterion = CombinedLoss()
    return model, criterion, device


def load_and_preprocess_data(
    train_noisy_path: str = "noisy_images_small_1k.npy",
    train_clean_path: str = "clean_images_small_1k.npy",
    test_noisy_path: str = "noisy_val_1k_harder.npy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data, normalizes to [0,1], casts to float32, and adds channel dim.

    Returns:
        noisy_train: (N, 1, 64, 64)
        clean_train: (N, 1, 64, 64)
        noisy_test:  (N, 1, 128, 128)
    """
    noisy_train = np.load(train_noisy_path).astype(np.float32) / 255.0
    clean_train = np.load(train_clean_path).astype(np.float32) / 255.0
    noisy_test = np.load(test_noisy_path).astype(np.float32) / 255.0

    noisy_train = np.expand_dims(noisy_train, axis=1)
    clean_train = np.expand_dims(clean_train, axis=1)
    noisy_test = np.expand_dims(noisy_test, axis=1)

    return noisy_train, clean_train, noisy_test


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for noisy_batch, clean_batch in loader:
        noisy_batch = noisy_batch.to(device, non_blocking=True)
        clean_batch = clean_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(noisy_batch)
        loss = criterion(pred, clean_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy_batch.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0

    for noisy_batch, clean_batch in loader:
        noisy_batch = noisy_batch.to(device, non_blocking=True)
        clean_batch = clean_batch.to(device, non_blocking=True)

        pred = model(noisy_batch)
        loss = criterion(pred, clean_batch)
        running_loss += loss.item() * noisy_batch.size(0)

    return running_loss / len(loader.dataset)


def run_kfold_training(
    noisy_train: np.ndarray,
    clean_train: np.ndarray,
    n_splits: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 50,
    save_dir: str = "checkpoints",
) -> Tuple[List[float], Dict[int, Dict[str, List[float]]]]:
    device = get_device()
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    os.makedirs(save_dir, exist_ok=True)

    best_val_losses: List[float] = []
    history: Dict[int, Dict[str, List[float]]] = {}

    pin_memory = device.type == "cuda"

    for fold, (train_idx, val_idx) in enumerate(kfold.split(noisy_train), start=1):
        print(f"\n===== Fold {fold}/{n_splits} =====")

        train_noisy = noisy_train[train_idx]
        train_clean = clean_train[train_idx]
        val_noisy = noisy_train[val_idx]
        val_clean = clean_train[val_idx]

        train_dataset = DenoisingDataset(train_noisy, train_clean, augment=True)
        val_dataset = DenoisingDataset(val_noisy, val_clean, augment=False)

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

        model = UNetDenoiser().to(device)
        criterion = CombinedLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        fold_train_losses: List[float] = []
        fold_val_losses: List[float] = []
        best_val_loss = float("inf")
        best_path = os.path.join(save_dir, f"best_unet_fold{fold}.pt")

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)

            print(
                f"Fold {fold} | Epoch {epoch:02d}/{num_epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)

        history[fold] = {
            "train_loss": fold_train_losses,
            "val_loss": fold_val_losses,
        }
        best_val_losses.append(best_val_loss)

        print(f"Best val_loss for fold {fold}: {best_val_loss:.6f}")
        print(f"Saved best weights to: {best_path}")

    return best_val_losses, history


def build_config_definition() -> Dict[str, Dict[str, object]]:
    return {
        "Config A": {
            "model_kwargs": {
                "dropout_p": 0.0,
                "use_batchnorm": False,
                "residual_encoder": False,
            },
            "loss": nn.MSELoss(),
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "scheduler": None,
            "augment": False,
            "color": "tab:blue",
        },
        "Config B": {
            "model_kwargs": {
                "dropout_p": 0.1,
                "use_batchnorm": True,
                "residual_encoder": False,
            },
            "loss": CombinedLoss(0.8, 0.2),
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "plateau",
            "augment": True,
            "color": "tab:orange",
        },
        "Config C": {
            "model_kwargs": {
                "dropout_p": 0.1,
                "use_batchnorm": True,
                "residual_encoder": True,
            },
            "loss": MSEPlusL1PlusSSIMLoss(0.6, 0.2, 0.2),
            "optimizer": "adamw",
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "augment": True,
            "color": "tab:green",
        },
    }


def run_config_comparison_experiment(
    noisy_train: np.ndarray,
    clean_train: np.ndarray,
    batch_size: int = 32,
    num_epochs: int = 50,
    save_plot_path: str = "config_comparison.png",
    save_dir: str = "config_checkpoints",
    seed: int = 42,
) -> Dict[str, Dict[str, List[float]]]:
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    indices = np.arange(noisy_train.shape[0])
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=seed, shuffle=True
    )

    config_defs = build_config_definition()
    all_history: Dict[str, Dict[str, List[float]]] = {}
    pin_memory = device.type == "cuda"

    for config_name, cfg in config_defs.items():
        print(f"\n===== {config_name} =====")

        train_dataset = DenoisingDataset(
            noisy_train[train_indices],
            clean_train[train_indices],
            augment=bool(cfg["augment"]),
        )
        val_dataset = DenoisingDataset(
            noisy_train[val_indices],
            clean_train[val_indices],
            augment=False,
        )

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

        model = UNetDenoiser(**cfg["model_kwargs"]).to(device)
        criterion = cfg["loss"]

        if cfg["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
            )

        scheduler = None
        if cfg["scheduler"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5
            )
        elif cfg["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        checkpoint_path = os.path.join(
            save_dir, config_name.lower().replace(" ", "_") + "_best.pt"
        )

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"{config_name} | Epoch {epoch:02d}/{num_epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), checkpoint_path)

        print(f"Best val_loss for {config_name}: {best_val:.6f}")
        print(f"Saved best weights to: {checkpoint_path}")
        all_history[config_name] = history

    plt.figure(figsize=(12, 7))
    for config_name, cfg in config_defs.items():
        color = str(cfg["color"])
        epochs = np.arange(1, len(all_history[config_name]["train_loss"]) + 1)
        plt.plot(
            epochs,
            all_history[config_name]["train_loss"],
            color=color,
            linestyle="-",
            label=f"{config_name} Train",
        )
        plt.plot(
            epochs,
            all_history[config_name]["val_loss"],
            color=color,
            linestyle="--",
            label=f"{config_name} Val",
        )

    plt.title("Configuration Comparison (Train Solid, Val Dashed)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=150)
    plt.close()

    print(f"Saved comparison figure to: {save_plot_path}")
    return all_history


def build_test_loader(noisy_test: np.ndarray, batch_size: int = 32) -> DataLoader:
    """Builds a test DataLoader from unlabeled noisy images."""
    test_dataset = DenoisingDataset(noisy_test, clean=None, augment=False)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def denoise_single_image(model: nn.Module, noisy_image_2d: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(noisy_image_2d).float().unsqueeze(0).unsqueeze(0).to(device)
    pred = model(x)
    return pred.squeeze(0).squeeze(0).cpu().numpy()


if __name__ == "__main__":
    noisy_train, clean_train, noisy_test = load_and_preprocess_data(
        train_noisy_path="noisy_images_small_1k.npy",
        train_clean_path="clean_images_small_1k.npy",
        test_noisy_path="noisy_val_1k_harder.npy",
    )

    print("Loaded train noisy shape:", noisy_train.shape)
    print("Loaded train clean shape:", clean_train.shape)
    print("Loaded test noisy shape:", noisy_test.shape)

    # Optional test loader for later inference on unlabeled 128x128 noisy set.
    test_loader = build_test_loader(noisy_test, batch_size=32)
    print("Test batches:", len(test_loader))

    run_config_comparison_experiment(
        noisy_train=noisy_train,
        clean_train=clean_train,
        batch_size=32,
        num_epochs=50,
        save_plot_path="config_comparison.png",
        save_dir="config_checkpoints",
        seed=42,
    )

    # Visualization model: use Config B architecture and load best checkpoint if present.
    device = get_device()
    viz_model = UNetDenoiser(dropout_p=0.1, use_batchnorm=True, residual_encoder=False).to(device)
    checkpoint_path = os.path.join("config_checkpoints", "config_b_best.pt")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        viz_model.load_state_dict(state_dict)
        print(f"Loaded visualization model weights from: {checkpoint_path}")
    else:
        print(
            "Warning: config_checkpoints/config_b_best.pt not found. "
            "Using current model weights for visualization."
        )

    # Training sample comparisons: indices 0 and 42.
    for idx in [0, 42]:
        noisy_img = noisy_train[idx, 0]
        clean_img = clean_train[idx, 0]
        denoised_img = denoise_single_image(viz_model, noisy_img, device)
        plot_image_comparison(
            noisy=noisy_img,
            clean=clean_img,
            denoised=denoised_img,
            title="Training Sample",
            filename=f"training_sample_{idx}.png",
        )

    # Test sample comparisons: indices 0 and 42 (no clean labels available).
    for idx in [0, 42]:
        noisy_img = noisy_test[idx, 0]
        denoised_img = denoise_single_image(viz_model, noisy_img, device)
        plot_image_comparison(
            noisy=noisy_img,
            clean=None,
            denoised=denoised_img,
            title="Test Sample",
            filename=f"test_sample_{idx}.png",
        )

    # First-layer filter visualization.
    plot_learned_filters(viz_model, filename="filters.png")

    # Optional k-fold plot utility call (uncomment if you run run_kfold_training).
    # best_losses, kfold_history = run_kfold_training(noisy_train, clean_train)
    # train_lists = [kfold_history[f]["train_loss"] for f in sorted(kfold_history)]
    # val_lists = [kfold_history[f]["val_loss"] for f in sorted(kfold_history)]
    # plot_kfold_results(train_lists, val_lists)



