import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Dice functions provided by the course (required) ----

def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    """Soft Dice loss (probabilities, used for training)."""
    probs = torch.sigmoid(logits)  # (B,1,H,W)
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def dice_score_from_logits(logits, targets, eps: float = 1e-6):
    """Hard Dice score (threshold 0.5, used for reporting)."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


@torch.no_grad()
def iou_score_from_logits(logits, targets, eps: float = 1e-6):
    """Hard IoU score, same style as dice_score_from_logits."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


class DiceBCELoss(nn.Module):
    """BCE handles per-pixel accuracy, Dice handles overlap / class imbalance."""

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return bce + dice_loss_from_logits(logits, target)


def denorm(img_tensor):
    """(1,H,W) normalized tensor -> (H,W) numpy in [0,1]."""
    return (img_tensor.squeeze(0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)


def overlay(ax, img, mask, color=(1, 0, 0), alpha=0.4):
    ax.imshow(img, cmap="gray")
    rgba = np.zeros((*mask.shape, 4))
    rgba[..., :3] = color
    rgba[..., 3] = mask * alpha
    ax.imshow(rgba)
    ax.axis("off")


def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss (BCE + Dice)")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["val_dice"], label="val Dice")
    axes[1].plot(epochs, history["val_iou"], label="val IoU")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
