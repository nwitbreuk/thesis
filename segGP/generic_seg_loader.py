import os
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MASK_EXTS = (".png", ".bmp")


def _find_files(root: str, exts: Tuple[str, ...]) -> Dict[str, str]:
    """Recursively index files under root by basename (without extension).

    Returns a dict: {stem: absolute_path}
    If multiple with same stem exist, the last one wins (undefined order).
    """
    out: Dict[str, str] = {}
    for r, _dirs, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                stem = os.path.splitext(fn)[0]
                out[stem] = os.path.join(r, fn)
    return out


class GeneralSegDataset(Dataset):
    """General-purpose segmentation dataset.

    Supports two label formats:
      - mode="png": per-pixel masks stored as images (PNG/BMP), matched by filename stem.
      - mode="yolo": YOLO txt bounding boxes per image, rasterized to a mask.

    Returns:
      - image: float tensor (C,H,W) in [0,1]
      - mask:
          if num_classes == 1: float tensor (1,H,W) in {0,1}
          else: long tensor (H,W) with class indices (0=background). Uses ignore_index=255 when present in masks.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        mode: str = "png",  # "png" | "yolo"
        color_mode: str = "gray",  # "rgb" | "gray"
        num_classes: int = 1,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.image_dir = os.path.abspath(image_dir)
        self.mask_dir = os.path.abspath(mask_dir) if mask_dir else None
        self.mode = mode.lower()
        self.color_mode = color_mode.lower()
        self.num_classes = int(max(1, num_classes))
        self.ignore_index = int(ignore_index)

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"image_dir not found: {self.image_dir}")

        # Index images and masks
        img_index = _find_files(self.image_dir, IMG_EXTS)
        if self.mode == "png":
            if self.mask_dir is None:
                # Allow masks to be co-located with images
                mask_index = _find_files(self.image_dir, MASK_EXTS)
            else:
                if not os.path.isdir(self.mask_dir):
                    raise FileNotFoundError(f"mask_dir not found: {self.mask_dir}")
                mask_index = _find_files(self.mask_dir, MASK_EXTS)
        elif self.mode == "yolo":
            # mask_dir defaults to image_dir if not provided
            label_root = self.mask_dir or self.image_dir
            mask_index = {k: v for k, v in _find_files(label_root, (".txt",)).items()}
        else:
            raise ValueError(f"Unknown mode '{self.mode}', expected 'png' or 'yolo'")

        # Build samples: only pairs with both image and label
        pairs: List[Tuple[str, str]] = []
        missing: List[str] = []
        for stem, img_path in img_index.items():
            if stem in mask_index:
                pairs.append((img_path, mask_index[stem]))
            else:
                missing.append(stem)

        if not pairs:
            raise RuntimeError(
                f"No (image, label) pairs found. Checked {len(img_index)} images under '{self.image_dir}' and labels under '{self.mask_dir or self.image_dir}'."
            )

        self.samples = pairs
        self.num_missing = len(missing)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        im = Image.open(path)
        if self.color_mode == "rgb":
            im = im.convert("RGB")
        else:
            im = im.convert("L")
        t = F.to_tensor(im)  # float [0,1], (C,H,W)
        return t

    def _rasterize_yolo(self, txt_path: str, size_hw: Tuple[int, int]) -> np.ndarray:
        H, W = size_hw
        if self.num_classes == 1:
            mask = np.zeros((H, W), dtype=np.uint8)
        else:
            mask = np.zeros((H, W), dtype=np.uint8)  # background=0
        try:
            with open(txt_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
        except Exception:
            return mask

        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
            except Exception:
                continue
            # YOLO normalized coords -> pixel box
            bw = int(round(w * W))
            bh = int(round(h * H))
            x = int(round((cx * W) - bw / 2))
            y = int(round((cy * H) - bh / 2))
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(W, x + bw)
            y2 = min(H, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            if self.num_classes == 1:
                mask[y1:y2, x1:x2] = 1
            else:
                # shift class ids by +1 so background stays 0
                cls = int(cid + 1)
                mask[y1:y2, x1:x2] = cls
        return mask

    def _load_mask_png(self, path: str, size_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
        m = Image.open(path)
        # Keep as indexed/label image; convert to numpy directly
        m_np = np.array(m)
        if m_np.ndim == 3:
            # If mask accidentally has 3 channels, collapse to single channel via first channel
            m_np = m_np[..., 0]
        # Ensure dtype small integer
        if m_np.dtype == np.bool_:
            m_np = m_np.astype(np.uint8)
        return m_np

    def __getitem__(self, idx: int):
        img_path, lab_path = self.samples[idx]
        img = self._load_image(img_path)
        _, H, W = img.shape

        if self.mode == "png":
            m_np = self._load_mask_png(lab_path, (H, W))
        else:
            m_np = self._rasterize_yolo(lab_path, (H, W))

        # Prepare tensor mask per configuration
        if self.num_classes == 1:
            # Binary: threshold non-zero -> 0/1, shape (1,H,W), float32
            m_bin = (m_np.astype(np.int64) != self.ignore_index) & (m_np.astype(np.int64) > 0)
            m_t = torch.from_numpy(m_bin.astype(np.float32)).unsqueeze(0)
        else:
            # Multiclass: keep as class indices, shape (H,W), long; 255 is ignore
            m_cls = m_np.astype(np.int64)
            m_t = torch.from_numpy(m_cls).long()

        return img, m_t

    # Provide attribute used by training loop/collate
    @property
    def ignore_index(self) -> int:  # type: ignore[override]
        return self._ignore_index

    @ignore_index.setter
    def ignore_index(self, val: int) -> None:
        self._ignore_index = int(val)
