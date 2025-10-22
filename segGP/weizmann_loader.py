import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from typing import Protocol

class _ResamplingLike(Protocol):
    NEAREST: int
    BILINEAR: int
    BICUBIC: int
    LANCZOS: int
    BOX: int
    HAMMING: int

try:
    from PIL.Image import Resampling as _PILResampling
    Resampling: _ResamplingLike = _PILResampling  # type: ignore[assignment]
except Exception:
    class _ResamplingCompat:
        NEAREST  = getattr(Image, "NEAREST", 0)
        LANCZOS  = getattr(Image, "LANCZOS", 1)
        BILINEAR = getattr(Image, "BILINEAR", 2)
        BICUBIC  = getattr(Image, "BICUBIC", 3)
        BOX      = getattr(Image, "BOX", 4)
        HAMMING  = getattr(Image, "HAMMING", 5)
    Resampling: _ResamplingLike = _ResamplingCompat()


class WeizmannHorseDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=None, extensions=(".png", ".jpg", ".jpeg")):
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.extensions = tuple(e.lower() for e in extensions)
        self.size = size

        print(f"Loading images from: {self.img_dir}")
        print(f"Loading masks from: {self.mask_dir}")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask dir not found: {self.mask_dir}")

        self.filenames = sorted([f for f in os.listdir(self.img_dir)
                                 if f.lower().endswith(self.extensions)])
        if not self.filenames:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        missing = [f for f in self.filenames if not os.path.exists(os.path.join(self.mask_dir, f))]
        if missing:
            raise FileNotFoundError(f"Missing masks for: {missing[:5]} (and more)")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")  # keep color
        msk = Image.open(os.path.join(self.mask_dir, fname)).convert("L")

        if self.size is not None:
            img = img.resize(self.size, resample=Resampling.BILINEAR)
            msk = msk.resize(self.size, resample=Resampling.NEAREST)

        img_np  = np.asarray(img, dtype=np.float32) / 255.0           # (H,W,3) in [0,1]
        img_np  = np.transpose(img_np, (2, 0, 1))                     # (3,H,W)
        mask_u8 = np.asarray(msk, dtype=np.uint8)
        mask_np = (mask_u8 > 0).astype(np.float32)           # (H,W)

        img_t  = torch.from_numpy(img_np)                    # (3,H,W)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)      # (1,H,W)
        return img_t, mask_t