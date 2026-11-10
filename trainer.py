import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from utils import DiceBCELoss, dice_score_from_logits, iou_score_from_logits, plot_history


class Trainer:
    def __init__(self, model, train_loader, val_loader, args, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device

        self.criterion = DiceBCELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5)

        self.run_dir = os.path.join(args.out_dir, args.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.best_path = os.path.join(self.run_dir, "best_model.pth")
        self.history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for img, mask, _ in pbar:
            img, mask = img.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(img), mask)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return float(np.mean(losses))

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        losses, dice_sum, iou_sum, n = [], 0.0, 0.0, 0
        for img, mask, _ in self.val_loader:
            img, mask = img.to(self.device), mask.to(self.device)
            logits = self.model(img)
            losses.append(self.criterion(logits, mask).item())
            b = img.size(0)  # weight by batch size -> mean over all images
            dice_sum += dice_score_from_logits(logits, mask) * b
            iou_sum += iou_score_from_logits(logits, mask) * b
            n += b
        return float(np.mean(losses)), dice_sum / n, iou_sum / n

    def fit(self):
        best_dice, best_epoch, bad_epochs = -1.0, 0, 0
        print(f"Training on {self.device} | train={len(self.train_loader.dataset)} "
              f"val={len(self.val_loader.dataset)} | epochs={self.args.epochs}")
        t0 = time.time()

        for epoch in range(1, self.args.epochs + 1):
            tr_loss = self.train_one_epoch(epoch)
            val_loss, val_dice, val_iou = self.validate()
            self.scheduler.step(val_dice)

            for k, v in zip(self.history, (tr_loss, val_loss, val_dice, val_iou)):
                self.history[k].append(v)

            mark = ""
            if val_dice > best_dice:
                best_dice, best_epoch, bad_epochs = val_dice, epoch, 0
                torch.save({"model": self.model.state_dict(), "epoch": epoch,
                            "val_dice": val_dice, "val_iou": val_iou,
                            "base_ch": self.args.base_ch}, self.best_path)
                mark = "  <- best, saved"
            else:
                bad_epochs += 1

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{self.args.epochs} | train_loss {tr_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_dice {val_dice:.4f} | val_iou {val_iou:.4f} | "
                  f"lr {lr:.1e}{mark}")

            if bad_epochs >= self.args.patience:
                print(f"Early stopping: no improvement for {self.args.patience} epochs.")
                break

        plot_history(self.history, os.path.join(self.run_dir, "training_curves.png"))
        with open(os.path.join(self.run_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Done in {time.time() - t0:.0f}s. Best val Dice {best_dice:.4f} at epoch {best_epoch}. "
              f"Saved to {self.best_path}")
