from tkinter.messagebox import IGNORE
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'   # <-- this defines device

def _to_imshow(img_t):
    """Convert a torch tensor image of shape (C,H,W) or (H,W[,C]) to a NumPy array
    acceptable by matplotlib.imshow, and return (array, cmap).
    - If 3-channel: returns (H,W,3), cmap=None
    - If 1-channel: returns (H,W), cmap='gray'
    """
    import numpy as _np
    import torch as _torch
    if isinstance(img_t, _torch.Tensor):
        arr = img_t.detach().cpu().numpy()
    else:
        arr = _np.asarray(img_t)

    cmap = None
    if arr.ndim == 2:
        # (H,W)
        cmap = 'gray'
        return arr, cmap
    if arr.ndim == 3:
        H, W = arr.shape[-2], arr.shape[-1]
        # CHW?
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] == H and arr.shape[2] == W:
            C = arr.shape[0]
            if C == 1:
                return arr[0], 'gray'
            return _np.transpose(arr, (1, 2, 0)), None
        # HWC?
        if arr.shape[2] in (1, 3, 4):
            C = arr.shape[2]
            if C == 1:
                return arr[:, :, 0], 'gray'
            return arr, None
        # Fallback: take first channel and show gray
        return arr[0], 'gray'
    # Fallback for unexpected dims: squeeze and recurse
    return _to_imshow(_np.squeeze(arr))

def _colorize_mask(mask_2d: torch.Tensor, k: int):
    """Colorize a 2D mask tensor of shape (H,W) with k classes into an RGB image.
    Assumes class ids in [0, k-1], with possible ignore index 255.
    Returns a (H,W,3) uint8 NumPy array.
    """   
    m = mask_2d.detach().cpu().numpy().astype(np.int32)
    h, w = m.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create palette for k classes (0..k-1)
    # We use a distinct colormap
    pal = plt.get_cmap('tab20', max(2, k))
    pal_rgb = (pal(np.arange(max(2, k)))[:, :3] * 255).astype(np.uint8)
    
    # Identify ignore regions (255)
    ignore_mask = (m == 255)
    valid_mask = ~ignore_mask
    
    # Color valid pixels
    # Clip only valid pixels to range [0, k-1] to be safe
    if valid_mask.any():
        idx = np.clip(m[valid_mask], 0, k - 1)
        out[valid_mask] = pal_rgb[idx]
        
    # Color ignore pixels as Gray
    out[ignore_mask] = np.array([192, 192, 192], dtype=np.uint8)
    
    return out

def visualize_predictions(toolbox, individual, dataset, num_samples=3, threshold=0.5, save_dir=None, num_classes=None):
    """
    Visualize predictions of a GP individual on samples from a dataset.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    func = toolbox.compile(expr=individual)
    idxs = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    for i, idx in enumerate(idxs):
        img, mask = dataset[idx]  # img: (C,H,W), mask: (1,H,W)
        with torch.no_grad():
            inp = img.unsqueeze(0).to(device)         # (1,C,H,W)
            out = func(inp)
            # Determine whether output is binary or multiclass by channel dim
            if out.dim() == 4 and out.shape[1] > 1:
                # multiclass: softmax then argmax
                probs = torch.softmax(out, dim=1)
                pred = probs.argmax(dim=1, keepdim=True).float()  # (1,1,H,W) as float for compatibility
            else:
                # binary: sigmoid + threshold
                probs = torch.sigmoid(out)
                pred = (probs > threshold).float()          # (1,1,H,W)
        # Prepare image for display (robust to CHW/HWC)
        img_np, img_cmap = _to_imshow(img)
        # prepare numpy arrays for display
        # mask may be (1,H,W) (binary) or (H,W) (multiclass long)
        if isinstance(mask, torch.Tensor):
            mask_t = mask.cpu()
        else:
            # if dataset returned numpy, convert
            mask_t = torch.from_numpy(np.asarray(mask))

        if mask_t.dim() == 3 and mask_t.shape[0] == 1:
            mask_np = mask_t.squeeze(0).numpy()
        else:
            mask_np = mask_t.numpy()

        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()

        # Choose display for multiclass vs binary
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(img_np, cmap=img_cmap)
        axs[0].set_title("Input")

        # infer num_classes if not provided
        ncls = None
        if num_classes is not None:
            ncls = int(num_classes)
        elif hasattr(dataset, 'num_classes'):
            try:
                ncls = int(getattr(dataset, 'num_classes'))
            except Exception:
                ncls = None

        if out is not None and out.dim() == 4 and out.shape[1] > 1:
            # multiclass display
            if ncls is None:
                ncls = int(out.shape[1])
            
            # Get the raw prediction indices (0 to k-1)
            pred_indices = pred.squeeze(0).squeeze(0)
            
            # Get the ground truth mask and find where the ignore_index is
            gt_mask_for_ignore = mask_t.squeeze(0) if mask_t.dim()==3 else mask_t
            ignore_pixels = (gt_mask_for_ignore == 255)

            # Create a new prediction mask where ignored pixels are set to 255
            pred_with_ignore = pred_indices.clone()
            pred_with_ignore[ignore_pixels] = 255

            pred_mc = _colorize_mask(pred_with_ignore, ncls)
            axs[1].imshow(pred_mc)
            axs[1].set_title("Prediction")

            mc = _colorize_mask(gt_mask_for_ignore, ncls)
            axs[2].imshow(mc); axs[2].set_title("Ground Truth - multiclass")
        else:
            # binary display
            axs[1].imshow(pred_np, cmap='gray')
            axs[1].set_title("Prediction")
            axs[2].imshow(mask_np, cmap='gray')
            axs[2].set_title("Ground Truth - binary")
        for ax in axs: ax.axis('off')
        fig.tight_layout()

        if save_dir:
            fig.savefig(os.path.join(save_dir, f"vis_{i}_idx{idx}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
    if not save_dir:
        plt.show()