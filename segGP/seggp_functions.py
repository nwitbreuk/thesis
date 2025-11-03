import torch
import torch.nn.functional as F   # <-- this defines F
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # <-- this defines device
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from typing import List
from collections import OrderedDict
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
def sigmoid(x): return torch.sigmoid(x)
def tanh_f(x): return torch.tanh(x)

# comparison and thresholding functions
def gt(x, t): return (x > t).float()
def lt(x, t): return (x < t).float()
def _restore_shape(y: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # Return to 2D/3D if input was 2D/3D
    if like.dim() == 2:      # (H,W)
        return y.squeeze(0).squeeze(0)
    if like.dim() == 3:      # (C,H,W)
        return y.squeeze(0)
    return y

# normalization and activation functions
def normalize(x: torch.Tensor) -> torch.Tensor:
    xin = x
    x = _as_nchw(x)
    B, C, H, W = x.shape
    mean = x.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    std = x.view(B, C, -1).std(dim=2).view(B, C, 1, 1) + 1e-6
    y = (x - mean) / std
    return _restore_shape(y, xin)
def clamp01(x): return torch.clamp(x, 0.0, 1.0)

# Logical operators
def logical_and(x, y): return (x * y)
def logical_or(x, y): return torch.clamp(x + y, 0, 1)
def logical_not(x): return 1 - x
def logical_xor(x, y): return torch.abs(x - y)



# endregion

# region ==== NN primitives ====

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, norm=True, activation=True):
    """Simple conv -> (BN) -> (ReLU) block. Returns an nn.Sequential module."""
    layers: List[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

# apply a fixed depthwise 3x3 kernel (stateless) -> behaves like a small conv filter
def apply_depthwise_edge(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Depthwise conv using a simple edge kernel scaled by `strength`. Stateless, no learnable params."""
    xin = x
    x = _as_nchw(x)
    B,C,H,W = x.shape
    # simple sharpening kernel (could be Sobel/laplacian variant)
    k = torch.tensor([[0., -1., 0.],
                      [-1., 4., -1.],
                      [0., -1., 0.]], device=x.device, dtype=x.dtype) * float(strength)
    k = k.unsqueeze(0).unsqueeze(0)   # (1,1,3,3)
    weight = k.repeat(C, 1, 1, 1)     # (C,1,3,3) => depthwise
    y = F.conv2d(x, weight, padding=1, groups=C)
    return _restore_like(y, xin)

# apply a small learnable conv module cached per (in_channels,out_channels)
_module_cache = OrderedDict()
_MAX_CACHE_SIZE = 64

# Small allowed set of channel choices to keep cache bounded and stable
_ALLOWED_CH = (8, 16, 32, 64, 128)

def _pick_channel(out_channels_f: float) -> int:
    """Map a GP-proposed float to a nearby allowed channel count."""
    try:
        v = int(round(float(out_channels_f)))
    except Exception:
        v = _ALLOWED_CH[0]
    return min(_ALLOWED_CH, key=lambda c: abs(c - v))
def apply_conv_block_cached(x: torch.Tensor, out_channels_f: float) -> torch.Tensor:
    """Apply a conv_block from input channels to out_channels (out_channels_f is float; cast to int).
    We cache nn.Module instances per (in_ch, out_ch) to avoid recreating each call.
    Modules are created on CPU then moved to input device when called.
    Note: cached modules are initialized once; they are used in inference only (no training in your pipeline)."""
    xin = x
    x = _as_nchw(x)
    B,C,H,W = x.shape
    out_channels = _pick_channel(out_channels_f)
    key = ("conv", C, out_channels, str(x.device))
    mod = _module_cache.get(key)
    if mod is None:
        mod = conv_block(C, out_channels, kernel_size=3, stride=1, norm=True, activation=True)
        # deterministic-ish initialization (per-process)
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
        _module_cache[key] = mod
        _module_cache.move_to_end(key)
        if len(_module_cache) > _MAX_CACHE_SIZE:
            _module_cache.popitem(last=False)
    # ensure module on same device/dtype
    mod = mod.to(x.device).to(x.dtype)
    y = mod(x)
    return _restore_like(y, xin)

def apply_aspp_cached(x: torch.Tensor, out_channels_f: float) -> torch.Tensor:
    """Apply a cached ASPP module. out_channels_f -> int out channels for ASPP projection."""
    xin = x
    x = _as_nchw(x)
    B,C,H,W = x.shape
    out_channels = _pick_channel(out_channels_f)
    key = ('aspp', C, out_channels, str(x.device))
    mod = _module_cache.get(key)
    if mod is None:
        mod = ASPP(C, out_channels, rates=(1,6,12))   # reduce rates for speed
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
        _module_cache[key] = mod
        _module_cache.move_to_end(key)
        if len(_module_cache) > _MAX_CACHE_SIZE:
            _module_cache.popitem(last=False)
    mod = mod.to(x.device).to(x.dtype)
    y = mod(x)
    return _restore_like(y, xin)


def apply_feature_to_mask_cached(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Map a feature-map tensor to single-channel logits (Mask). `scale` can modulate output magnitude.

    Signature: (FeatureMap, float) -> Mask
    """
    xin = x
    x = _as_nchw(x)
    B, C, H, W = x.shape
    key = ('feat2mask', C, str(x.device))
    mod = _module_cache.get(key)
    if mod is None:
        # lightweight 1x1 classifier to collapse channels -> 1
        mod = nn.Sequential(
            nn.Conv2d(C, 1, kernel_size=1, bias=False),
        )
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
        _module_cache[key] = mod
        _module_cache.move_to_end(key)
        if len(_module_cache) > _MAX_CACHE_SIZE:
            _module_cache.popitem(last=False)
    mod = mod.to(x.device).to(x.dtype)
    y = mod(x) * float(scale)
    return _restore_like(y, xin)


def ensure_rgb(x: torch.Tensor) -> torch.Tensor:
    """Ensure the tensor is RGB-like (3 channels). If single-channel, replicate; if >3, take first 3."""
    xin = x
    x = _as_nchw(x)
    if x.shape[1] == 3:
        return _restore_like(x, xin)
    if x.shape[1] == 1:
        return _restore_like(x.repeat(1,3,1,1), xin)
    # if more than 3 channels, truncate
    return _restore_like(x[:, :3, :, :], xin)


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB to single-channel gray using a simple luminosity rule."""
    xin = x
    x = _as_nchw(x)
    if x.shape[1] >= 3:
        r,g,b = x[:,0:1], x[:,1:2], x[:,2:3]
        y = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        y = x.mean(dim=1, keepdim=True)
    return _restore_like(y, xin)


def image_to_featuremap(x: torch.Tensor, ch_f: float) -> torch.Tensor:
    """High-level wrapper: ensure RGB, then apply cached conv_block to produce a feature map.

    Signature: (RGBImage, float) -> FeatureMap
    """
    xin = x
    x = ensure_rgb(x)
    return apply_conv_block_cached(x, ch_f)
class ASPP(nn.Module):
    """Lightweight Atrous Spatial Pyramid Pooling (ASPP)-like module.

    Usage: aspp = ASPP(in_channels, out_channels); y = aspp(x)
    """
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        self.rate_convs = nn.ModuleList()
        for r in rates:
            if r == 1:
                # 1x1 conv
                self.rate_convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ))
            else:
                self.rate_convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ))

        # global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d((len(rates) + 1) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.rate_convs:
            res.append(conv(x))
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=res[0].shape[2:], mode='bilinear', align_corners=False)
        res.append(gp)
        cat = torch.cat(res, dim=1)
        return self.project(cat)

def simple_mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU):
    """Return a simple MLP (nn.Sequential) with given hidden dims.

    Example: simple_mlp(512, [256,128], 10)
    """
    layers = []
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)

def simple_fcn(in_channels, num_classes):
    """A 1x1 classifier conv producing raw logits for segmentation classes."""
    return nn.Conv2d(in_channels, num_classes, kernel_size=1)

def parameter_count(model: nn.Module):
    """Return (total_params, trainable_params) as integers for the given model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)

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
# endregion


