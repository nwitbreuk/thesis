import torch
import torch.nn.functional as F   # <-- this defines F
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # <-- this defines device
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from typing import List
from collections import OrderedDict
import math
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

# region ==== Combination & alignment Helpers ====

def _project_channels_cached(x: torch.Tensor, out_ch: int) -> torch.Tensor:
    """Project tensor channels to `out_ch` via a cached 1x1 conv (deterministic init, inference-only).

    x: (H,W), (C,H,W) or (B,C,H,W) tensor
    returns: same batch/spatial shape with C=out_ch
    """
    xin = x
    x = _as_nchw(x)
    B, C, H, W = x.shape
    if C == out_ch:
        return _restore_like(x, xin)
    key = ("proj1x1", C, out_ch, str(x.device))
    mod = _module_cache.get(key)
    if mod is None:
        mod = nn.Conv2d(C, out_ch, kernel_size=1, bias=False)
        # deterministic-ish init
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5)) if "math" in globals() else nn.init.kaiming_uniform_(p)
        _module_cache[key] = mod
        _module_cache.move_to_end(key)
        if len(_module_cache) > _MAX_CACHE_SIZE:
            _module_cache.popitem(last=False)
    mod = mod.to(x.device).to(x.dtype)
    y = mod(x)
    return _restore_like(y, xin)

def _repeat_channels(x: torch.Tensor, target_c: int) -> torch.Tensor:
    xin = x
    x = _as_nchw(x)
    B, C, H, W = x.shape
    if C == target_c:
        return _restore_like(x, xin)
    if C == 1:
        y = x.repeat(1, target_c, 1, 1)
        return _restore_like(y, xin)
    # fallback to projection when C>1 and unequal
    return _project_channels_cached(x, target_c)

def _align_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align two tensors on channel dimension for safe elementwise ops.
    Rules:
      - If channels equal: passthrough
      - If one is single-channel: repeat to the other's channels
      - Else: project both down to min(Ca, Cb) using cached 1x1 conv
    """
    ain, bin = a, b
    a4 = _as_nchw(a)
    b4 = _as_nchw(b)
    Ca = a4.shape[1]
    Cb = b4.shape[1]
    if Ca == Cb:
        return a, b
    if Ca == 1 and Cb > 1:
        return _repeat_channels(a, Cb), b
    if Cb == 1 and Ca > 1:
        return a, _repeat_channels(b, Ca)
    # project both to the smaller channel count to reduce cost
    target = min(Ca, Cb)
    a_al = _project_channels_cached(a, target)
    b_al = _project_channels_cached(b, target)
    return a_al, b_al

def mix(x, y, w: float):
    """Convex combination of two tensors with automatic channel alignment."""
    x_al, y_al = _align_pair(x, y)
    return x_al * w + y_al * (1.0 - w)

def if_then_else(cond, a, b):
    """Select between a and b based on condition tensor, aligning channels as needed."""
    a_al, b_al = _align_pair(a, b)
    # cond to 1 channel via mean, then repeat to match target channels
    cond4 = _as_nchw(cond)
    Ctarget = _as_nchw(a_al).shape[1]
    if cond4.shape[1] != 1:
        cond4 = cond4.mean(dim=1, keepdim=True)
    cond_al = _repeat_channels(cond4, Ctarget)
    return cond_al * a_al + (1.0 - cond_al) * b_al

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
# region ==== Basic Math operators (channel-safe) ====

def add(x, y):
    xa, ya = _align_pair(x, y)
    return xa + ya

def sub(x, y):
    xa, ya = _align_pair(x, y)
    return xa - ya

def mul(x, y):
    xa, ya = _align_pair(x, y)
    return xa * ya

def safe_div(x, y):
    xa, ya = _align_pair(x, y)
    return xa / (ya + 1e-6)
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

# Logical operators (channel-safe)
def logical_and(x, y):
    xa, ya = _align_pair(x, y)
    return xa * ya

def logical_or(x, y):
    xa, ya = _align_pair(x, y)
    return torch.clamp(xa + ya, 0, 1)

def logical_not(x):
    return 1 - x

def logical_xor(x, y):
    xa, ya = _align_pair(x, y)
    return torch.abs(xa - ya)



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
        # Initialize weights
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
        # Use eval() to avoid BatchNorm complaining about batch-size=1 (GP often runs per-image)
        mod.eval()
        _module_cache[key] = mod
        _module_cache.move_to_end(key)
        if len(_module_cache) > _MAX_CACHE_SIZE:
            _module_cache.popitem(last=False)
    else:
        # Ensure in eval mode even when fetched from cache
        mod.eval()
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
    - Returns feature maps (logits) as (same batch shape, 32, H, W) to be combined downstream.

    Note: Previously returned 21-class logits. To treat the NN as a feature extractor
    rather than a full segmenter, we reduce the channel dimensionality via a 1x1 conv
    to a compact feature representation; decoding to classes is left to the GP tree.
    """
    xin = x
    x = _as_nchw(x).to(device)          # (B,C,H,W)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        x = x[:, :3]

    x = (x - _MEAN) / _STD
    out = _PRETRAINED(x)["out"]         # (B,21,H,W)
    # Project to a compact feature map to discourage direct class output dominance
    key = ("pre_nn_proj", out.shape[1], 32, str(out.device))
    mod = _module_cache.get(key)
    if mod is None:
        mod = nn.Conv2d(out.shape[1], 32, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(mod.weight)
        _module_cache[key] = mod
        _module_cache.move_to_end(key)
        if len(_module_cache) > _MAX_CACHE_SIZE:
            _module_cache.popitem(last=False)
    mod = mod.to(out.device).to(out.dtype)
    feats = mod(out)
    return _restore_like(feats, xin)

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

# region ==== Pretrained feature combinators ====

def combine_pre_feat(pre_feats: torch.Tensor, fmap: torch.Tensor, w: float) -> torch.Tensor:
    """
    Combine pretrained features with another feature map using a convex mix after channel alignment.

    Args:
        pre_feats: tensor-like features produced by pretrained_seg_nn (any shape convertible to NCHW)
        fmap: tensor-like regular feature map
        w: mixing coefficient in [0,1]; output = w * A + (1-w) * B

    Returns: Feature map tensor aligned to input spatial dims.
    """
    # Reuse existing helpers to align and mix safely
    # mix(x, y, w) already aligns channels and returns x*w + y*(1-w)
    return mix(pre_feats, fmap, float(max(0.0, min(1.0, w))))

# endregion


