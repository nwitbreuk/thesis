import torch
import torch.nn.functional as F   # <-- this defines F
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # <-- this defines device
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch.nn as nn


# Helpers
def _as_nchw(x: torch.Tensor) -> torch.Tensor:
    # Accept (H,W), (C,H,W) or (B,C,H,W); return (B,C,H,W)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)         # (1,1,H,W)
    elif x.dim() == 3:
        x = x.unsqueeze(0)                       # (1,C,H,W)
    elif x.dim() != 4:
        raise ValueError(f"Expected 2D/3D/4D, got shape {tuple(x.shape)}")
    return x

def _restore_like(y: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # Return to 2D/3D if input was 2D/3D
    if like.dim() == 2:      # (H,W)
        return y.squeeze(0).squeeze(0)
    if like.dim() == 3:      # (C,H,W)
        return y.squeeze(0)
    return y   

# region ==== Combination Functions ====

def mix(x, y, w: float):
    """"
    Mix two tensors with a weight.
    Args:
        x: Tensor 1.
        y: Tensor 2.
        w: Weight for tensor 1, in [0,1].
    Returns:
        Mixed tensor.
    """
    # convex combination, w in [0,1]
    return x * w + y * (1.0 - w)

def if_then_else(cond, a, b):
    """
    Select between two tensors based on a condition tensor.
    Args:
        cond: Condition tensor, expected to be in {0,1}.
        a: Tensor if condition is true (1).
        b: Tensor if condition is false (0).
    Returns:
        Selected tensor.
    """
    # cond expected in {0,1}; broadcast select
    return cond * a + (1.0 - cond) * b

def gaussian_blur_param(x, sigma: float):
    xin = x
    x = _as_nchw(x)
    B, C, H, W = x.shape
    sigma = float(max(0.5, min(5.0, sigma)))
    k = max(3, 2 * int(3 * sigma) + 1)
    ax = torch.arange(-k // 2 + 1., k // 2 + 1., device=x.device)
    g1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1 = g1 / g1.sum()
    k2 = torch.outer(g1, g1).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)  # (C,1,k,k)
    y = F.conv2d(x, k2, padding=k // 2, groups=C)
    return _restore_like(y, xin)

# endregion
# region ==== Basic Math operators ====

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def safe_div(x, y): return x / (y + 1e-6)
def abs_f(x): return torch.abs(x)
def sqrt_f(x): return torch.sqrt(torch.clamp(x, min=0))
def log_f(x): return torch.log1p(torch.abs(x))
def exp_f(x): return torch.exp(torch.clamp(x, max=10))

# normalization and activation functions
def normalize(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min + 1e-6)

def sigmoid(x): return torch.sigmoid(x)
def tanh_f(x): return torch.tanh(x)

# comparison and thresholding functions
def gt(x, t): return (x > t).float()
def lt(x, t): return (x < t).float()
def clamp01(x): return torch.clamp(x, 0.0, 1.0)

# Logical operators
def logical_and(x, y): return (x * y)
def logical_or(x, y): return torch.clamp(x + y, 0, 1)
def logical_not(x): return 1 - x
def xor(x, y): return torch.abs(x - y)



# endregion

# region ==== NN primitives ====


# Load and freeze a pre-trained segmentation model (DeepLabV3 with ResNet50 backbone)
# Pre-trained on COCO; outputs 21 classes, but we'll use class 18 (horse) or adapt for binary
# Load weights with proper meta (silences deprecated warnings)
_WEIGHTS = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
_PRETRAINED = deeplabv3_resnet50(weights=_WEIGHTS).to(device).eval()
for p in _PRETRAINED.parameters():
    p.requires_grad_(False)

_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
_CATS = _WEIGHTS.meta.get("categories", [])
_HORSE_IDX = (_CATS.index("horse") if "horse" in _CATS else 13)  # VOC index for 'horse' fallback

@torch.no_grad()
def pretrained_seg_nn(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts (H,W), (1,H,W), (3,H,W) or (B,C,H,W).
    - If grayscale (C=1), replicate to 3 channels.
    - Normalize with weights meta.
    - Returns logits for 'horse' as (same batch shape, 1, H, W).
    """
    xin = x
    x = _as_nchw(x).to(device)          # (B,C,H,W)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        x = x[:, :3]

    x = (x - _MEAN) / _STD
    out = _PRETRAINED(x)["out"]         # (B,21,H,W)
    horse_logits = out[:, _HORSE_IDX:_HORSE_IDX+1]  # (B,1,H,W)
    return _restore_like(horse_logits, xin)




# endregion

# region ==== Feature Extraction Functions ====

# endregion

# region ==== Filtering & edge detection Functions ====

# Sobel and Laplacian filters
sobel_x_kernel = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=torch.float32, device=device)
sobel_y_kernel = torch.tensor([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]], dtype=torch.float32, device=device)
laplacian_kernel = torch.tensor([[0, 1, 0],
                                 [1,-4, 1],
                                 [0, 1, 0]], dtype=torch.float32, device=device)

def _depthwise_conv(x: torch.Tensor, k2d: torch.Tensor, pad: int) -> torch.Tensor:
    xin = x
    x = _as_nchw(x)
    B, C, H, W = x.shape
    k = k2d.to(x.device)
    if k.dim() == 2:
        k = k.unsqueeze(0).unsqueeze(0)        # (1,1,kh,kw)
    weight = k.repeat(C, 1, 1, 1)              # (C,1,kh,kw)
    y = F.conv2d(x, weight, padding=pad, groups=C)
    return _restore_like(y, xin)

def sobel_x(x):   return _depthwise_conv(x, sobel_x_kernel, pad=1)
def sobel_y(x):   return _depthwise_conv(x, sobel_y_kernel, pad=1)
def laplacian(x): return _depthwise_conv(x, laplacian_kernel, pad=1)

def gradient_magnitude(x):
    gx, gy = sobel_x(x), sobel_y(x)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


