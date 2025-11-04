import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from deap import base, gp
from weizmann_loader import WeizmannHorseDataset

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

def visualize_predictions(toolbox, individual, dataset, num_samples=3, threshold=0.5, save_dir=None):
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
            out = torch.sigmoid(func(inp))            # (1,1,H,W) expected
            pred = (out > threshold).float()          # (1,1,H,W)
        # Prepare image for display (robust to CHW/HWC)
        img_np, img_cmap = _to_imshow(img)
        mask_np = mask.squeeze(0).cpu().numpy()
        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(img_np, cmap=img_cmap)
        axs[0].set_title("Input")
        axs[1].imshow(pred_np, cmap='gray')
        axs[1].set_title("Prediction")
        axs[2].imshow(mask_np, cmap='gray')
        axs[2].set_title("Ground Truth")
        for ax in axs: ax.axis('off')
        fig.tight_layout()

        if save_dir:
            fig.savefig(os.path.join(save_dir, f"vis_{i}_idx{idx}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
    if not save_dir:
        plt.show()