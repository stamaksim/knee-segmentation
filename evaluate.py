import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import get_loader
from model import UNet
from utils import dice_score_from_logits, iou_score_from_logits, denorm, overlay


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = UNet(base=ckpt.get("base_ch", 32)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {path} (epoch {ckpt['epoch']}, val Dice at save time {ckpt['val_dice']:.4f})")
    return model


@torch.no_grad()
def evaluate_labeled(model, loader, device, threshold, save_path, n_show=6):
    """Dice/IoU on a split that has ground-truth masks (val) + GT vs prediction figure."""
    dices, ious, examples = [], [], []
    for img, mask, names in loader:
        logits = model(img.to(device)).cpu()
        # per-image scores with the course's Dice function
        d = [dice_score_from_logits(logits[k:k + 1], mask[k:k + 1]) for k in range(len(names))]
        i = [iou_score_from_logits(logits[k:k + 1], mask[k:k + 1]) for k in range(len(names))]
        dices += d
        ious += i
        pred = (torch.sigmoid(logits) > threshold).float()
        for k in range(len(names)):
            if len(examples) < n_show:
                examples.append((names[k], img[k], mask[k], pred[k], d[k]))

    print(f"Validation ({len(dices)} images): Dice {np.mean(dices):.4f} ± {np.std(dices):.4f} | "
          f"IoU {np.mean(ious):.4f} ± {np.std(ious):.4f}")

    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
    for r, (name, img, mask, pred, d) in enumerate(examples):
        im = denorm(img)
        axes[r, 0].imshow(im, cmap="gray"); axes[r, 0].axis("off"); axes[r, 0].set_title(name)
        overlay(axes[r, 1], im, mask[0].numpy(), color=(0, 1, 0)); axes[r, 1].set_title("ground truth")
        overlay(axes[r, 2], im, pred[0].numpy(), color=(1, 0, 0)); axes[r, 2].set_title(f"prediction (Dice {d:.3f})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    print(f"Saved figure: {save_path}")
    return float(np.mean(dices)), float(np.mean(ious))


@torch.no_grad()
def predict_test(model, loader, device, threshold, out_dir, fig_path, n_show=8):
    """Test set has no masks: save predicted masks as PNG + an overlay figure."""
    mask_dir = os.path.join(out_dir, "test_predictions")
    os.makedirs(mask_dir, exist_ok=True)
    examples = []
    for img, _, names in loader:
        pred = (torch.sigmoid(model(img.to(device))) > threshold).cpu().numpy().astype(np.uint8)
        for k, name in enumerate(names):
            Image.fromarray(pred[k, 0] * 255).save(os.path.join(mask_dir, name))
            if len(examples) < n_show:
                examples.append((name, img[k], pred[k, 0]))
    print(f"Saved {len(loader.dataset)} predicted test masks to {mask_dir}/")

    cols = 4
    rows = int(np.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, img, pred) in zip(axes.flat, examples):
        overlay(ax, denorm(img), pred, color=(1, 0, 0))
        ax.set_title(name)
    fig.suptitle("Test set: predicted masks")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


def run_evaluation(args, device):
    model = load_model(args.checkpoint, device)
    out_dir = os.path.dirname(args.checkpoint)

    val_loader = get_loader(args.csv_dir, "val", args.batch_size, train=False,
                            num_workers=args.num_workers, img_size=args.img_size)
    test_loader = get_loader(args.csv_dir, "test", args.batch_size, train=False,
                             num_workers=args.num_workers, img_size=args.img_size)

    evaluate_labeled(model, val_loader, device, args.threshold,
                     os.path.join(out_dir, "val_examples.png"))
    predict_test(model, test_loader, device, args.threshold, out_dir,
                 os.path.join(out_dir, "test_examples.png"))
