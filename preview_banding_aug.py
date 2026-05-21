"""
Visual comparison of banding augmentation applied to real noisy training images.

Columns: Original noisy | Augmented (train input) | Test ref 42
SSIM vs clean is shown per cell for the first two columns.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_msssim import ssim as torch_ssim

RNG_SEED = 0


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def aug_v2(img: np.ndarray, clean: np.ndarray, rng: random.Random) -> np.ndarray:
    img = img.copy()
    _, h, w = img.shape
    for _ in range(rng.randint(30, 50)):
        is_dark  = rng.random() < 0.7
        is_horiz = rng.random() < 0.5
        width    = max(1, min(10, int(rng.expovariate(1.0))))
        max_d    = min(0.25 / pow(width, 1.5), 0.12)
        delta    = rng.uniform(max_d * 0.4, max_d)
        sign     = -1 if is_dark else 1
        if is_horiz:
            pos = rng.randint(0, h - width)
            img[:, pos:pos + width, :] += sign * delta
        else:
            pos = rng.randint(0, w - width)
            img[:, :, pos:pos + width] += sign * delta
    if rng.random() < 0.3:
        spacing = rng.randint(10, 25)
        delta   = rng.uniform(0.05, 0.10)
        for pos in range(0, h, spacing):
            img[:, pos:pos + 1, :] -= delta
        for pos in range(0, w, spacing):
            img[:, :, pos:pos + 1] -= delta
    return img.clip(0, 1)


def compute_ssim(clean: np.ndarray, img: np.ndarray) -> float:
    c = torch.from_numpy(clean).unsqueeze(0)
    i = torch.from_numpy(img).unsqueeze(0)
    return float(torch_ssim(i, c, data_range=1.0, size_average=True).item())


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

noisy_train = np.load("data/noisy_train_19k_harder.npy", allow_pickle=True).astype(np.float32) / 255.0
clean_train = np.load("data/clean_train_19k_harder.npy", allow_pickle=True).astype(np.float32) / 255.0
noisy_test  = np.load("data/noisy_val_1k_harder.npy",  allow_pickle=True).astype(np.float32) / 255.0

noisy_train = np.expand_dims(noisy_train, axis=1)
clean_train = np.expand_dims(clean_train, axis=1)
noisy_test  = np.expand_dims(noisy_test,  axis=1)

train_indices = [7, 13, 42, 77]
test_ref_idx  = 42

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

n_rows = len(train_indices)
n_cols = 3  # original | augmented | test ref

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
fig.suptitle(
    "Banding augmentation preview  (applied ON TOP of existing noisy training images)",
    fontsize=10,
)

axes[0, 0].set_title("Original noisy", fontsize=9)
axes[0, 1].set_title("Augmented (train input)", fontsize=9)
axes[0, 2].set_title(f"Test ref (idx={test_ref_idx})", fontsize=9)

test_ref = noisy_test[test_ref_idx]

for row, train_idx in enumerate(train_indices):
    orig  = noisy_train[train_idx]
    clean = clean_train[train_idx]
    aug   = aug_v2(orig, clean, random.Random(RNG_SEED + row))

    orig_ssim = compute_ssim(clean, orig)
    aug_ssim  = compute_ssim(clean, aug)

    imgs    = [orig,      aug,      test_ref]
    titles  = [f"SSIM={orig_ssim:.3f}", f"SSIM={aug_ssim:.3f}", ""]

    for col, (img, subtitle) in enumerate(zip(imgs, titles)):
        ax = axes[row, col]
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(f"train idx {train_idx}", fontsize=8)
        if subtitle:
            ax.set_xlabel(subtitle, fontsize=8)

plt.tight_layout()
out = "banding_aug_proposal_preview.png"
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved → {out}")
