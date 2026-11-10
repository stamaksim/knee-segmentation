import os
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

# masks are palette PNGs with transparency; PIL warns on convert("L") but the result is correct
warnings.filterwarnings("ignore", message="Palette images with Transparency")


def get_transforms(train):
    if not train:
        return None
    # geometric ops are applied to image AND mask, ColorJitter only to the image
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ])


class KneeDataset(Dataset):
    """Reads (xray, mask) paths from a CSV. Mask column may be 'None' (unlabeled test set)."""

    def __init__(self, csv_path, train=False, img_size=224):
        self.df = pd.read_csv(csv_path, keep_default_na=False)
        self.transform = get_transforms(train)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def _load(self, path, is_mask):
        img = Image.open(path)
        # masks are palette PNGs: convert('L') maps the palette to 0/255
        img = img.convert("L")
        if img.size != (self.img_size, self.img_size):
            resample = Image.NEAREST if is_mask else Image.BILINEAR
            img = img.resize((self.img_size, self.img_size), resample)
        return np.array(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = torch.from_numpy(self._load(row["xrays"], False)).unsqueeze(0)  # (1,H,W) uint8
        has_mask = row["masks"] not in ("None", "", None)
        if has_mask:
            mask = torch.from_numpy((self._load(row["masks"], True) > 127).astype(np.uint8)).unsqueeze(0)
        else:
            mask = torch.zeros_like(img)

        img, mask = tv_tensors.Image(img), tv_tensors.Mask(mask)
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        img = (img.float() / 255.0 - 0.5) / 0.5      # -> [-1, 1]
        mask = mask.float()
        name = os.path.basename(row["xrays"])
        return img.as_subclass(torch.Tensor), mask.as_subclass(torch.Tensor), name


def get_loader(csv_dir, split, batch_size, train, num_workers=2, img_size=224):
    ds = KneeDataset(os.path.join(csv_dir, f"{split}.csv"), train=train, img_size=img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())
