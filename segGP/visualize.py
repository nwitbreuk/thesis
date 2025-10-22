import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from deap import base, gp
from weizmann_loader import WeizmannHorseDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'   # <-- this defines device

def _to_numpy_img(img_t):
    arr = img_t.cpu().numpy()  # (C,H,W)
    if arr.ndim == 3 and arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0)), None   # (H,W,3)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0], 'gray'                       # (H,W)
    return arr, 'gray'

def visualize_predictions(toolbox, individual, dataset, num_samples=3, threshold=0.5, save_dir=None):
    """
    Visualize predictions of a GP individual on samples from a dataset.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    func = toolbox.compile(expr=individual)
    idxs = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    for i, idx in enumerate(idxs):
        img, mask = dataset[idx]                  # img: (3,H,W) or (1,H,W), mask: (1,H,W)
        with torch.no_grad():
            inp  = img.unsqueeze(0).to(device)    # (1,C,H,W)
            out  = func(inp)                      # (1,C',H,W)
            if out.dim() == 4 and out.shape[1] > 1:
                out = out.mean(dim=1, keepdim=True)
            probs = torch.sigmoid(out)            # (1,1,H,W)
            pred  = (probs > threshold).float()   # (1,1,H,W)

        img_np, cmap_input = _to_numpy_img(img)
        mask_np = mask.squeeze(0).cpu().numpy()
        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(img_np, cmap=cmap_input)
        axs[0].set_title("Input")
        axs[1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axs[1].set_title("Prediction")
        axs[2].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axs[2].set_title("Ground Truth")
        for ax in axs: ax.axis('off')
        fig.tight_layout()

        if save_dir:
            fig.savefig(os.path.join(save_dir, f"vis_{i}_idx{idx}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()