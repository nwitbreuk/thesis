import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= CONFIGURATION =================
IMAGE_DIR = "/dataB1/aoi/benchmarks/ensemble_datasets/log_furex/train/img"
MASK_DIR = "/dataB1/aoi/benchmarks/ensemble_datasets/log_furex/train/lbl"

MODEL_PATHS = {
    "aoi_1": "/dataB1/aoi/benchmarks/model_library/ensemble_models/v2/ensemble_model_1.pth",
    "aoi_2": "/dataB1/aoi/benchmarks/model_library/ensemble_models/v2/ensemble_model_2.pth",
    "aoi_3": "/dataB1/aoi/benchmarks/model_library/ensemble_models/v2/ensemble_model_3.pth",
}

NUM_CLASSES = 5
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

# ================= UTILS =================

def decode_binary_mask(label: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Decodes a binary mask from an integer-encoded label in numpy (using bit unpacking)."""
    label = np.array(label, dtype=np.uint8)
    # Unpack bits: expands last dim into 8 bits, we take the first num_classes
    # Input shape: (H, W) -> Output shape: (H, W, 8) -> Slice: (H, W, num_classes)
    label = np.unpackbits(label[..., None], axis=-1, bitorder="little")[..., :num_classes]
    return label

def calculate_iou(pred_mask, true_mask, eps=1e-7):
    """
    Calculates IoU per class for binary masks (B, C, H, W).
    Uses the metric: sum(intersection[valid] / union[valid]) / sum(iou_valid)
    Only counts classes that have GT pixels or predictions.
    """
    # Convert to boolean for bitwise operations
    pred_mask = pred_mask.bool().float()
    true_mask = true_mask.bool().float()
    
    # Compute intersection and union per class
    # Shape: (C,)
    intersection = (true_mask * pred_mask).sum(dim=(0, 2, 3))
    union = ((true_mask + pred_mask) > 0).sum(dim=(0, 2, 3)).float()
    
    # acc_valid: classes with GT pixels
    acc_valid = true_mask.sum(dim=(0, 2, 3)) > 0
    
    # iou_valid: classes with either GT or predictions
    iou_valid = (true_mask.sum(dim=(0, 2, 3)) + pred_mask.sum(dim=(0, 2, 3))) > 0
    
    # Avoid division by zero
    iou_per_class = torch.zeros_like(intersection)
    valid_union = union > 0
    iou_per_class[valid_union] = intersection[valid_union] / union[valid_union]
    
    # Return per-class IoU (will aggregate in main loop)
    return iou_per_class, acc_valid, iou_valid

def visualize_sample(image, gt_mask, pred_mask, name, save_path=None):
    """
    Visualize Image, GT (Multi-label) and Prediction (Multi-label).
    Expects masks in shape (C, H, W).
    """
    # Prepare Image (take first channel of dual lighting for vis)
    img_vis = image[0].cpu().numpy()
    img_vis = (img_vis * 255).astype(np.uint8)  # Normalize from [0,1] to [0,255]
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2RGB)

    # Helper to create RGB overlay from multi-channel mask
    def create_overlay(mask_tensor):
        mask_np = mask_tensor.cpu().numpy()  # (C, H, W)
        h, w = mask_np.shape[1], mask_np.shape[2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color map for up to 5 classes
        colors = [
            (255, 0, 0),     # Class 0: Red
            (0, 255, 0),     # Class 1: Green
            (0, 0, 255),     # Class 2: Blue
            (255, 255, 0),   # Class 3: Cyan
            (255, 0, 255)    # Class 4: Magenta
        ]
        
        for c in range(min(len(colors), mask_np.shape[0])):
            m = mask_np[c] > 0
            color = colors[c]
            # Additive blending for overlap visualization
            overlay[m] = np.clip(overlay[m] + np.array(color, dtype=np.uint8), 0, 255)
            
        return overlay

    gt_overlay = create_overlay(gt_mask)
    pred_overlay = create_overlay(pred_mask)

    # Blend with original image
    alpha = 0.6
    gt_vis = cv2.addWeighted(img_vis, 1 - alpha, gt_overlay, alpha, 0)
    pred_vis = cv2.addWeighted(img_vis, 1 - alpha, pred_overlay, alpha, 0)

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_vis, cmap="gray")
    ax[0].set_title(f"Input Image (Buffer00)\n{name}")
    ax[0].axis('off')
    
    ax[1].imshow(gt_vis)
    ax[1].set_title("Ground Truth (Index decoded)")
    ax[1].axis('off')
    
    ax[2].imshow(pred_vis)
    ax[2].set_title(f"Prediction (Thresh {THRESHOLD})")
    ax[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()
    plt.close()

# ================= EXTRA CONFIG FOR DECODING =================
DECODE_SOURCE = "G_BITS"  # Use bit-unpacking like training
BITORDER = "little"
NUM_BITS_TO_DECODE = 5  # decode 5 LSBs from G channel

def decode_binary_mask_from_g(mask_rgb: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Decode multi-label mask from G channel LSBs (matches training encoding)."""
    g = mask_rgb[..., 1].astype(np.uint8)
    # Unpack LSBs: (H, W) -> (H, W, 8) -> take first num_classes
    bits = np.unpackbits(g[..., None], axis=-1, bitorder="little")[..., :num_classes]
    return bits

def decode_index_mask(mask_rgb: np.ndarray, num_classes: int):
    """If labels are index-encoded per pixel, convert to one-hot. Use G channel by default."""
    index = mask_rgb[..., 1].astype(np.int64)  # change channel if needed
    # Clamp indices to [0, num_classes-1]
    index = np.clip(index, 0, num_classes - 1)
    onehot = np.eye(num_classes, dtype=np.uint8)[index]  # (H,W,C)
    return onehot

def inspect_label_encoding(mask_dir: Path, sample_limit: int = 5):
    """Inspect RGB channel stats and bit usage to detect encoding mismatches."""
    print("\n[INSPECT] Scanning label encoding...")
    valid_extensions = {".bmp", ".png", ".jpg"}
    files = [f for f in mask_dir.iterdir() if f.suffix.lower() in valid_extensions][:sample_limit]
    if not files:
        print("[INSPECT] No mask files found.")
        return

    total_counts_rgb = {"R": {}, "G": {}, "B": {}}
    bit_counts_big = np.zeros(24, dtype=np.int64)
    bit_counts_little = np.zeros(24, dtype=np.int64)

    for f in files:
        m_img = Image.open(f).convert("RGB")
        m = np.array(m_img)
        for i, ch in enumerate(["R", "G", "B"]):
            vals, cnts = np.unique(m[..., i], return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                total_counts_rgb[ch][v] = total_counts_rgb[ch].get(v, 0) + c

        # Bit usage (big)
        big = np.concatenate([
            np.unpackbits(m[..., 0][..., None], axis=-1, bitorder="big"),
            np.unpackbits(m[..., 1][..., None], axis=-1, bitorder="big"),
            np.unpackbits(m[..., 2][..., None], axis=-1, bitorder="big"),
        ], axis=-1)  # (H,W,24)
        bit_counts_big += big.sum(axis=(0, 1), dtype=np.int64)

        # Bit usage (little)
        little = np.concatenate([
            np.unpackbits(m[..., 0][..., None], axis=-1, bitorder="little"),
            np.unpackbits(m[..., 1][..., None], axis=-1, bitorder="little"),
            np.unpackbits(m[..., 2][..., None], axis=-1, bitorder="little"),
        ], axis=-1)
        bit_counts_little += little.sum(axis=(0, 1), dtype=np.int64)

    print("[INSPECT] Unique values per channel (top 20):")
    for ch in ["R", "G", "B"]:
        items = sorted(total_counts_rgb[ch].items(), key=lambda x: -x[1])[:20]
        print(f"  {ch}: {items}")

    print("[INSPECT] Bit usage (big-endian) counts per 24 bits:")
    print(f"  {bit_counts_big.tolist()}")
    print("[INSPECT] Bit usage (little-endian) counts per 24 bits:")
    print(f"  {bit_counts_little.tolist()}")

    # Heuristic: if only a few distinct values exist in a single channel and each pixel has <=1 bit on, it's likely index-encoded.
    # If multiple bits per pixel are often on, it's multi-label bit-pack.
    # You can use these counts to select BITORDER and which bits map to your 5 classes.

# ================= DATASET =================

def _collect_dual_pairs(img_dir: Path, mask_dir: Path):
    pairs = []
    # Get all valid image names
    valid_extensions = {".bmp", ".png", ".jpg"}
    all_imgs = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in valid_extensions}
    
    for mfile in mask_dir.iterdir():
        if mfile.suffix.lower() not in valid_extensions:
            continue
            
        # Naming convention assumption: Label name matches Image name (buffer00)
        # Adjust logic if label name differs slightly from image name
        stem = mfile.stem
        
        # Try to find corresponding buffer00 image
        img0_name = stem
        img1_name = stem.replace("buffer00", "buffer01")
        
        if img0_name not in all_imgs:
             # Try fallback if mask name doesn't contain buffer00 but image does
             pass 

        if img0_name in all_imgs:
            img0_path = all_imgs[img0_name]
            # Find buffer01 or reuse buffer00
            img1_path = all_imgs.get(img1_name, img0_path)
            pairs.append((img0_path, img1_path, mfile))
            
    return pairs

class DualLightingDataset(Dataset):
    def __init__(self, pairs, num_classes=NUM_CLASSES):
        self.pairs = pairs
        self.num_classes = num_classes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img0_path, img1_path, mask_path = self.pairs[idx]
        i0 = np.array(Image.open(img0_path).convert("L"), dtype=np.float32) / 255.0
        i1 = np.array(Image.open(img1_path).convert("L"), dtype=np.float32) / 255.0
        img = torch.from_numpy(np.stack([i0, i1], axis=0)).float()

        m_img = Image.open(mask_path).convert("RGB")
        mask_np = np.array(m_img)

        # Use the same decoding as training (bit-unpacking from G channel)
        binary_mask_np = decode_binary_mask_from_g(mask_np, num_classes=self.num_classes)

        mask = torch.from_numpy(binary_mask_np.transpose(2, 0, 1)).float()
        return img, mask, img0_path.stem

# ================= METRICS (safe mIoU over present classes) =================
def calculate_miou_over_present(avg_iou: torch.Tensor, gt_pixels: torch.Tensor):
    """Average IoU only over classes that have any GT pixels."""
    present = gt_pixels > 0
    if present.sum() == 0:
        return 0.0
    return avg_iou[present].mean().item()

# ================= MAIN EVALUATION =================
def run_evaluation():
    img_dir_path = Path(IMAGE_DIR)
    mask_dir_path = Path(MASK_DIR)

    # Inspect GT encoding first
    inspect_label_encoding(mask_dir_path, sample_limit=3)

    print(f"[Data] Scanning pairs in {img_dir_path}...")
    pairs = _collect_dual_pairs(img_dir_path, mask_dir_path)
    print(f"[Data] Found {len(pairs)} valid pairs.")
    if len(pairs) == 0:
        print("No pairs found. Check paths.")
        return

    dataset = DualLightingDataset(pairs, num_classes=NUM_CLASSES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n{'='*20} Evaluating {model_name} {'='*20}")
        print(f"Loading: {model_path}")
        try:
            model = torch.jit.load(model_path, map_location=DEVICE)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue

        model.to(DEVICE)
        model.eval()

        total_iou_intersection = torch.zeros(NUM_CLASSES).to(DEVICE)
        total_iou_union = torch.zeros(NUM_CLASSES).to(DEVICE)
        total_acc_valid = torch.zeros(NUM_CLASSES).to(DEVICE)
        total_iou_valid = torch.zeros(NUM_CLASSES).to(DEVICE)
        
        gt_class_pixels = torch.zeros(NUM_CLASSES).to(DEVICE)
        pred_class_pixels = torch.zeros(NUM_CLASSES).to(DEVICE)
        n_samples = 0
        
        vis_indices = [0, len(pairs)//2, len(pairs)-1]
        vis_count = 0

        with torch.no_grad():
            for i, (images, masks, names) in enumerate(tqdm(dataloader)):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                probs = torch.sigmoid(outputs)
                preds = (probs > THRESHOLD).float()

                if i == 0:
                    print(f"\n[DEBUG] First batch diagnostics:")
                    print(f"  Images shape: {images.shape}")
                    print(f"  GT masks shape: {masks.shape}")
                    print(f"  Model output shape: {outputs.shape}")
                    print(f"  Predictions shape: {preds.shape}")
                    print(f"  GT unique values: {torch.unique(masks)}")
                    print(f"  Pred unique values: {torch.unique(preds)}")
                    print(f"  GT pixel counts per class: {masks.sum(dim=(0, 2, 3))}")
                    print(f"  Pred pixel counts per class: {preds.sum(dim=(0, 2, 3))}")

                gt_class_pixels += masks.sum(dim=(0, 2, 3))
                pred_class_pixels += preds.sum(dim=(0, 2, 3))

                # Calculate IoU components
                iou_per_class, acc_valid, iou_valid = calculate_iou(preds, masks)
                total_iou_intersection += iou_per_class * (iou_valid.float())
                total_iou_valid += iou_valid.float()
                total_acc_valid += acc_valid.float()
                
                n_samples += 1
                
                # Visualize 3 samples
                if vis_count < 3 and i in vis_indices:
                    save_path = f"viz_{model_name}_{names[0]}.png"
                    visualize_sample(images[0], masks[0], preds[0], names[0], save_path=save_path)
                    vis_count += 1
        
        # Final Metrics: sum(intersection[valid] / union[valid]) / sum(iou_valid)
        miou_all = (total_iou_intersection.sum() / total_iou_valid.sum()).item() if total_iou_valid.sum() > 0 else 1.0
        
        # For per-class: intersection / union where valid
        per_class_iou = torch.zeros(NUM_CLASSES).to(DEVICE)
        for c in range(NUM_CLASSES):
            if total_iou_valid[c] > 0:
                per_class_iou[c] = total_iou_intersection[c] / total_iou_valid[c]
        
        # mIoU only over classes present in GT (acc_valid)
        miou_present_only = per_class_iou[total_acc_valid > 0].mean().item() if (total_acc_valid > 0).sum() > 0 else 1.0

        print(f"\n{'='*60}")
        print(f"[RESULTS] Model: {model_name}")
        print(f"{'='*60}")
        print(f"  mIoU (all 5 classes): {miou_all:.4f}")
        print(f"  mIoU (only classes present in GT): {miou_present_only:.4f}")
        print(f"\n  Per-class IoU:")
        for c in range(NUM_CLASSES):
            if total_iou_valid[c] > 0:
                present_marker = " [in GT]" if total_acc_valid[c] > 0 else " [not in GT]"
                print(f"    Class {c}: {per_class_iou[c].item():.4f}{present_marker}")
            else:
                print(f"    Class {c}: N/A (no valid samples)")
        print(f"\n  Total GT pixels per class across dataset:")
        for c in range(NUM_CLASSES):
            print(f"    Class {c}: {int(gt_class_pixels[c].item())}")
        print(f"\n  Total Pred pixels per class across dataset:")
        for c in range(NUM_CLASSES):
            print(f"    Class {c}: {int(pred_class_pixels[c].item())}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    run_evaluation()

