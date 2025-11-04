import os
from typing import Optional, Sequence, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# PIL Resampling compatibility (Pillow>=9 vs older)
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

try:
    import torchvision
    from torchvision.datasets import VOCSegmentation
except Exception as e:  # pragma: no cover
    torchvision = None
    VOCSegmentation = None


class PascalVOCDataset(Dataset):
    """
    Thin wrapper around torchvision.datasets.VOCSegmentation that returns:
      - image tensor: (3,H,W) for rgb or (1,H,W) for gray, float32 in [0,1]
      - mask tensor: (H,W) long with class ids (0..20), ignore_index=255 preserved

    Args:
        root: Root directory that contains VOCdevkit/VOC2012 (or VOC2007).
        year: "2012" or "2007".
        image_set: "train", "val" or "trainval".
        download: Whether to download via torchvision (requires internet and write perms).
        color_mode: "rgb" or "gray" for returned image tensor channels.
        size: Optional (W,H) resize. Uses bilinear for image, nearest for mask.
        classes: Optional explicit number of classes to report (default 21 for VOC).
    """

    CLASS_NAMES_2012: List[str] = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        color_mode: str = "rgb",
        size: Optional[Tuple[int, int]] = None,
        classes: Optional[int] = None,
    ) -> None:
        if VOCSegmentation is None:
            raise ImportError("torchvision is required for PascalVOCDataset; install torchvision.")
        self.root = root
        self.year = year
        self.image_set = image_set
        self.ds = VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        # normalize color mode
        self.color_mode = (color_mode or "rgb").lower()
        self.size = size
        self.ignore_index = 255
        self.num_classes = int(classes or 21)
        self.class_names = self.CLASS_NAMES_2012[: self.num_classes] if self.num_classes == 21 else None

        print(f"[VOC] Using VOC{year} split='{image_set}' from root: {root}")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img_pil, target_pil = self.ds[idx]  # PIL.Image.Image, PIL.Image.Image
        # Convert target to numpy int64 with ignore_index preserved
        target_np = np.array(target_pil, dtype=np.uint8)
        # Resize if requested
        if self.size is not None:
            # PIL expects (width, height)
            w, h = self.size
            # Image bilinear, mask nearest
            img_pil = img_pil.resize((w, h), resample=Resampling.BILINEAR)
            target_pil = target_pil.resize((w, h), resample=Resampling.NEAREST)
            target_np = np.array(target_pil, dtype=np.uint8)

        # Convert image to float32 [0,1]
        if self.color_mode == "rgb":
            img_pil = img_pil.convert("RGB")
            img_np = np.asarray(img_pil, dtype=np.float32) / 255.0  # (H,W,3)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)       # (3,H,W)
        else:
            img_pil = img_pil.convert("L")
            img_np = np.asarray(img_pil, dtype=np.float32) / 255.0  # (H,W)
            img_t = torch.from_numpy(img_np).unsqueeze(0)           # (1,H,W)

        # Mask as (H,W) long. Keep ignore_index=255.
        mask_t = torch.from_numpy(target_np.astype(np.int64))       # (H,W)

        return img_t, mask_t

    # Convenience props
    @property
    def classes(self) -> int:
        return self.num_classes

    @property
    def ignore(self) -> int:
        return self.ignore_index

