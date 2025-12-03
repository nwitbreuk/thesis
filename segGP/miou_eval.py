"""
Dataset-level mIoU evaluation helpers for segGP.

This module provides utilities to accumulate a pixel-level confusion matrix
across a DataLoader and compute per-class IoU and mean IoU (mIoU).

It also provides `eval_dataset_miou(...)` which compiles an individual via the
provided DEAP toolbox, runs it across a DataLoader (train/test/val), accumulates
confusion counts, and returns per-class IoU and mIoU. It does NOT change any
existing code or integrate into `segGP_main.py`; it's provided as a reviewable
implementation you can plug in.

Usage (example):

from miou_eval import eval_dataset_miou
mious = eval_dataset_miou(toolbox, individual, test_loader, num_classes=21, device='cpu', ignore_index=255)

"""
from typing import Optional, Tuple
import numpy as np
import torch


def _accumulate_confmat(conf: np.ndarray, preds_np: np.ndarray, target_np: np.ndarray, num_classes: int, ignore_index: Optional[int] = None) -> np.ndarray:
    """Accumulate confusion matrix counts.

    conf: (num_classes, num_classes) ndarray to add into (true, pred)
    preds_np, target_np: 1D numpy arrays of same length
    ignore_index: if given, pixels with target == ignore_index are skipped
    Returns the updated conf (the same array object is modified).
    """
    # build mask to select valid pixels
    mask = np.ones_like(target_np, dtype=bool)
    if ignore_index is not None:
        mask &= (target_np != ignore_index)
    # ensure labels are within [0, num_classes-1]
    mask &= (target_np >= 0) & (target_np < num_classes)
    mask &= (preds_np >= 0) & (preds_np < num_classes)
    if mask.sum() == 0:
        return conf
    preds_np = preds_np[mask]
    target_np = target_np[mask]
    # encode pair (t, p) -> single index
    k = (target_np * num_classes + preds_np).astype(np.int64)
    binc = np.bincount(k, minlength=num_classes * num_classes)
    conf += binc.reshape((num_classes, num_classes)).astype(np.int64)
    return conf


def confmat_to_iou(conf: np.ndarray, exclude_background: bool = True) -> Tuple[np.ndarray, float]:
    """Compute per-class IoU and mean IoU from confusion matrix.

    conf: (C,C) ndarray where conf[true, pred] = pixel counts
    exclude_background: if True and C>1, exclude class 0 from mean mIoU
    Returns: (ious_array_of_length_C, mean_iou)
    """
    C = conf.shape[0]
    ious = np.full((C,), np.nan, dtype=float)
    for c in range(C):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious[c] = float(tp / denom)
        else:
            ious[c] = np.nan
    if exclude_background and C > 1:
        vals = ious[1:]
    else:
        vals = ious
    if np.all(np.isnan(vals)):
        miou = 0.0
    else:
        miou = float(np.nanmean(vals))
    return ious, miou


def _to_numpy_preds_and_targets(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert prediction and target tensors to 1D numpy arrays of class ids.

    preds: torch.Tensor logits or probabilities or discrete preds.
    targets: torch.Tensor ground-truth (H,W) or (B,H,W) or (B,1,H,W)

    Returns (pred_flat, target_flat) as 1D numpy arrays.
    """
    # preds expected: (B,C,H,W) or (B,1,H,W) or (C,H,W) etc.
    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets)

    # ensure batch dim
    if preds.dim() == 3:
        preds = preds.unsqueeze(0)
    if targets.dim() == 3:
        targets = targets.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)

    # If multiclass logits/probs -> argmax
    if preds.dim() == 4 and preds.shape[1] > 1:
        C = preds.shape[1]
        # align channels to expected num_classes to keep argmax indices in range
        if C != num_classes:
            if C > num_classes:
                preds = preds[:, :num_classes, ...]
            else:
                # pad missing channels with very negative logits so they won't win argmax
                pad_shape = (preds.shape[0], num_classes - C, preds.shape[2], preds.shape[3])
                pad = torch.full(pad_shape, -1e9, device=preds.device, dtype=preds.dtype)
                preds = torch.cat([preds, pad], dim=1)
        pred_ids = preds.argmax(dim=1)
    else:
        # binary: convert to single-channel {0,1}
        # ensure logits -> probs
        if preds.dim() == 4 and preds.shape[1] == 1:
            probs = torch.sigmoid(preds)
        else:
            probs = preds
        pred_ids = (probs > 0.5).long().squeeze(1)

    # target may be (B,1,H,W) for binary or (B,H,W) for multiclass
    if targets.dim() == 4 and targets.shape[1] == 1:
        targ_ids = targets.squeeze(1).long()
    else:
        targ_ids = targets.long()

    # --- KEY FIX: Align spatial dimensions before flattening ---
    # Ensure pred_ids and targ_ids have identical shapes
    if pred_ids.shape != targ_ids.shape:
        # Both should be (B, H, W) at this point
        # Interpolate predictions to match target spatial size
        target_h, target_w = targ_ids.shape[-2:]
        if pred_ids.dim() == 3:
            # Add channel dimension for interpolation: (B, H, W) -> (B, 1, H, W)
            pred_ids_4d = pred_ids.unsqueeze(1).float()
            pred_ids_4d = torch.nn.functional.interpolate(
                pred_ids_4d, 
                size=(target_h, target_w), 
                mode='nearest'
            )
            pred_ids = pred_ids_4d.squeeze(1).long()
        elif pred_ids.dim() == 2:
            # (H, W) case - add batch and channel dims
            pred_ids_4d = pred_ids.unsqueeze(0).unsqueeze(0).float()
            pred_ids_4d = torch.nn.functional.interpolate(
                pred_ids_4d,
                size=(target_h, target_w),
                mode='nearest'
            )
            pred_ids = pred_ids_4d.squeeze(0).squeeze(0).long()

    # flatten
    pred_flat = pred_ids.clamp(0, max(0, num_classes - 1)).cpu().numpy().ravel().astype(np.int64)
    targ_flat = targ_ids.cpu().numpy().ravel().astype(np.int64)
    
    # Final safety check: ensure arrays have same length
    if pred_flat.shape[0] != targ_flat.shape[0]:
        min_len = min(pred_flat.shape[0], targ_flat.shape[0])
        pred_flat = pred_flat[:min_len]
        targ_flat = targ_flat[:min_len]
    
    return pred_flat, targ_flat


def eval_dataset_miou(
        toolbox, 
        individual, 
        data_loader, 
        num_classes: int,  
        device: Optional[str] = None, 
        ignore_index: Optional[int] = 255, 
        max_batches: Optional[int] = None)-> Tuple[np.ndarray, float]:
    """Evaluate an individual across a DataLoader and return dataset-level IoUs.

    Args:
      toolbox: DEAP toolbox with `compile` registered
      individual: GP individual (PrimitiveTree)
      data_loader: DataLoader yielding (images, masks)
      num_classes: number of classes including background
      device: device string for inference (e.g. 'cuda' or 'cpu')
      ignore_index: label value to ignore (e.g. 255)
      max_batches: if set, limit number of batches to process (useful for fast tests)

    Returns:
      (ious_array_of_length_num_classes, mean_iou)

    Notes:
      - Does not apply any complexity/pretrained penalty. That can be applied by
        the caller using `len(individual)` or a helper provided here.
      - Runs under torch.no_grad and moves tensors to device for speed.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    func = toolbox.compile(expr=individual)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    processed = 0
    with torch.no_grad():
        for batch_i, (imgs, masks) in enumerate(data_loader):
            if max_batches is not None and batch_i >= max_batches:
                break
            imgs = imgs.to(device)
            # masks may be float (1,H,W) for binary or long (H,W) for multiclass
            out = func(imgs)
            # Convert outputs and masks to 1D numpy arrays of ids
            pred_flat, targ_flat = _to_numpy_preds_and_targets(out, masks, num_classes, ignore_index)
            # accumulate confusion
            _accumulate_confmat(conf, pred_flat, targ_flat, num_classes, ignore_index)
            processed += 1

    ious, miou = confmat_to_iou(conf, exclude_background=(num_classes > 1))
    return ious, miou


# Small helpers for penalties (caller can use these)

def complexity_penalty(individual, lambda_size: float = 1e-3) -> float:
    """Return penalty proportional to tree size. Default lambda small."""
    size = len(individual)
    return lambda_size * float(size)


def pretrained_usage_penalty(individual, primitive_name: str = 'PretrainedSeg', lambda_pre: float = 0.05) -> float:
    """Return penalty proportional to the number of occurrences of `primitive_name` in the tree."""
    cnt = 0
    try:
        for node in individual:
            # node may be gp.Primitive or Terminal; Primitive has .name
            name = getattr(node, 'name', None)
            if name == primitive_name:
                cnt += 1
    except Exception:
        pass
    return lambda_pre * float(cnt)
