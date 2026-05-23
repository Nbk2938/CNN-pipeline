from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from pytorch_msssim import ssim as torch_ssim
except ImportError:
    torch_ssim = None

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:
    skimage_ssim = None


def _compute_mse(clean: np.ndarray, candidate: np.ndarray) -> float:
    diff = clean.astype(np.float32) - candidate.astype(np.float32)
    return float(np.mean(diff * diff))


def _compute_ssim(clean: np.ndarray, candidate: np.ndarray) -> float:
    if torch_ssim is not None:
        clean_t = torch.from_numpy(clean).float().unsqueeze(0).unsqueeze(0)
        cand_t = torch.from_numpy(candidate).float().unsqueeze(0).unsqueeze(0)
        return float(torch_ssim(cand_t, clean_t, data_range=1.0, size_average=True).item())

    if skimage_ssim is not None:
        return float(skimage_ssim(clean, candidate, data_range=1.0))

    return float("nan")


def plot_kfold_results(fold_train_losses: List[List[float]], fold_val_losses: List[List[float]]):
    """
    Plot 5-fold train/val curves with thin fold lines and bold mean lines.
    Saves figure as kfold_curves.png.
    """
    train_arr = np.array(fold_train_losses, dtype=np.float32)
    val_arr = np.array(fold_val_losses, dtype=np.float32)
    epochs = np.arange(1, train_arr.shape[1] + 1)

    plt.figure(figsize=(12, 7))

    for fold in range(train_arr.shape[0]):
        plt.plot(epochs, train_arr[fold], color="tab:blue", alpha=0.3, linewidth=1.0)
        plt.plot(epochs, val_arr[fold], color="tab:orange", alpha=0.3, linewidth=1.0)

    plt.plot(
        epochs,
        train_arr.mean(axis=0),
        color="tab:blue",
        linewidth=3.0,
        label="Mean Train",
    )
    plt.plot(
        epochs,
        val_arr.mean(axis=0),
        color="tab:orange",
        linewidth=3.0,
        label="Mean Val",
    )

    plt.title("K-Fold Cross-Validation Loss (5 Folds)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("kfold_curves.png", dpi=150)
    plt.close()


def plot_image_comparison(
    noisy: np.ndarray,
    clean: Optional[np.ndarray],
    denoised: np.ndarray,
    title: str,
    filename: str,
    augmented: Optional[np.ndarray] = None,
    aug_clean: Optional[np.ndarray] = None,
):
    """
    Plot clean/noisy/augmented/denoised for labeled samples, noisy/denoised for unlabeled.
    augmented: one sample of the augmented noisy input the model actually trains on.
    aug_clean: the correspondingly transformed clean — used for augmented SSIM/MSE.
    Adds MSE/SSIM in titles where clean is available.
    """
    if clean is not None:
        noisy_mse  = _compute_mse(clean, noisy)
        noisy_ssim = _compute_ssim(clean, noisy)
        den_mse    = _compute_mse(clean, denoised)
        den_ssim   = _compute_ssim(clean, denoised)

        n_cols = 4 if augmented is not None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5))

        axes[0].imshow(clean, cmap="gray")
        axes[0].set_title("Clean")

        axes[1].imshow(noisy, cmap="gray")
        axes[1].set_title(f"Noisy (original)\nMSE={noisy_mse:.5f}, SSIM={noisy_ssim:.4f}")

        if augmented is not None:
            ref = aug_clean if aug_clean is not None else clean
            aug_mse  = _compute_mse(ref, augmented)
            aug_ssim = _compute_ssim(ref, augmented)
            axes[2].imshow(augmented, cmap="gray")
            axes[2].set_title(f"Augmented (train input)\nMSE={aug_mse:.5f}, SSIM={aug_ssim:.4f}")
            axes[3].imshow(denoised, cmap="gray")
            axes[3].set_title(f"Denoised\nMSE={den_mse:.5f}, SSIM={den_ssim:.4f}")
        else:
            axes[2].imshow(denoised, cmap="gray")
            axes[2].set_title(f"Denoised\nMSE={den_mse:.5f}, SSIM={den_ssim:.4f}")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(noisy, cmap="gray")
    axes[0].set_title("Noisy")
    axes[1].imshow(denoised, cmap="gray")
    axes[1].set_title("Denoised")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


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
