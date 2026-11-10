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


class DiceBCELoss(nn.Module):
    """BCE handles per-pixel accuracy, Dice handles overlap / class imbalance."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return bce + (1 - dice.mean())


@torch.no_grad()
def batch_metrics(logits, target, threshold=0.5, eps=1e-7):
    """Returns per-image Dice and IoU (lists) for a batch."""
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    p_sum = pred.sum(dim=(1, 2, 3))
    t_sum = target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (p_sum + t_sum + eps)
    iou = (inter + eps) / (p_sum + t_sum - inter + eps)
    return dice.tolist(), iou.tolist()


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
