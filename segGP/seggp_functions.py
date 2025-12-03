import torch
import torch.nn.functional as F   # <-- this defines F
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # <-- this defines device
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from typing import Any, Callable
from collections import OrderedDict
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import os

# region ==== Alignment & Helpers functions ====

def _fmt_arg(a: Any) -> str:
    try:
        if isinstance(a, torch.Tensor):
            return f"Tensor{tuple(a.shape)} dtype={a.dtype} dev={a.device}"
        return f"{type(a).__name__}({str(a)[:64]})"
    except Exception:
        return f"{type(a).__name__}"

def trace_primitive(name: str, fn: Callable) -> Callable:
    def _wrapped(*args, **kw):
        try:
            return fn(*args, **kw)
        except Exception as e:
            info = ", ".join(_fmt_arg(a) for a in args)
            raise RuntimeError(f"[Primitive {name}] {e.__class__.__name__}: {e}. Args: {info}") from e
    _wrapped.__name__ = f"traced_{name}"
    return _wrapped

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
    """Align two tensors on channel AND spatial dimensions for safe elementwise ops.
    Rules:
      - Align channels first (repeat single-channel or project to common size)
      - Then align spatial dimensions (interpolate to common H,W)
    """
    ain, bin = a, b
    a4 = _as_nchw(a)
    b4 = _as_nchw(b)
    Ca = a4.shape[1]
    Cb = b4.shape[1]
    Ha, Wa = a4.shape[2], a4.shape[3]
    Hb, Wb = b4.shape[2], b4.shape[3]
    
    # Step 1: Align channels
    if Ca != Cb:
        if Ca == 1 and Cb > 1:
            a4 = a4.repeat(1, Cb, 1, 1)
            Ca = Cb
        elif Cb == 1 and Ca > 1:
            b4 = b4.repeat(1, Ca, 1, 1)
            Cb = Ca
        else:
            # project both to the smaller channel count to reduce cost
            target = min(Ca, Cb)
            a4 = _project_channels_cached(a4, target)
            b4 = _project_channels_cached(b4, target)
    
    # Step 2: Align spatial dimensions - use the EXACT size from tensor 'a' as reference
    # This ensures deterministic alignment in nested operations like IfElse
    if (Ha, Wa) != (Hb, Wb):
        # Always align b to match a's spatial size
        b4 = F.interpolate(b4, size=(Ha, Wa), mode='bilinear', align_corners=False)
    
    # Restore original dimensionality (2D/3D/4D)
    a_aligned = _restore_like(a4, ain)
    b_aligned = _restore_like(b4, bin)
    
    return a_aligned, b_aligned

# endregion

# region ==== Combination functions ====

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

def feature_to_logits(x: torch.Tensor, k: int) -> torch.Tensor:
    """Project a feature map to k-class logits via cached 1x1 conv."""
    xin = x
    x4 = _as_nchw(x)
    B, C, H, W = x4.shape
    key = ('feat2logits', C, int(k), str(x4.device))
    mod = _module_cache.get(key)
    if mod is None:
        mod = nn.Conv2d(C, int(k), kernel_size=1, bias=False)
        nn.init.kaiming_normal_(mod.weight)
        _module_cache[key] = mod
        _module_cache.move_to_end(key)
        if len(_module_cache) > _MAX_CACHE_SIZE:
            _module_cache.popitem(last=False)
    mod = mod.to(x4.device).to(x4.dtype)
    y = mod(x4)
    return _restore_like(y, xin)

# endregion ==== Model Utilities ====

# region ==== NN models ====

# Load and freeze a pre-trained segmentation model (DeepLabV3 with ResNet50 backbone)
# Pre-trained on COCO; outputs 21 classes, but we'll use class 18 (horse) or adapt for binary
# Load weights with proper meta (silences deprecated warnings)
_WEIGHTS = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
# Lazy-load the pretrained model to avoid import-time downloads or failures
_PRETRAINED = None
def _get_pretrained():
    """Return the cached pretrained DeepLabV3 model, loading it on first use.

    Loading can attempt network access (or heavy disk IO); doing this at import
    time can cause the module import to fail and leave attributes undefined.
    Use lazy loading so importing the module is safe.
    """
    global _PRETRAINED
    if _PRETRAINED is None:
        _PRETRAINED = deeplabv3_resnet50(weights=_WEIGHTS).to(device).eval()
        for p in _PRETRAINED.parameters():
            p.requires_grad_(False)
    return _PRETRAINED

_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

@torch.no_grad()
def pretrained_seg_nn(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts (H,W), (1,H,W), (3,H,W) or (B,C,H,W).
    - If grayscale (C=1), replicate to 3 channels.
    - Normalize with weights meta.
    - Returns segmentation logits (same batch shape, 21, H, W) directly from the
      pretrained DeepLabV3 head. This restores the original behavior where the
      network acts as a full segmenter rather than only a feature extractor.
    
    For binary setups (NUM_CLASSES == 1) downstream code may still apply a
    channel selection or projection; here we always return the full 21-class
    logits to keep the primitive generic.
    """
    xin = x
    x = _as_nchw(x).to(device)          # (B,C,H,W)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        x = x[:, :3]

    x = (x - _MEAN) / _STD
    model = _get_pretrained()
    out = model(x)["out"]         # (B,21,H,W) VOC/COCO-aligned logits
    return _restore_like(out, xin)

@torch.no_grad()
def nn_feature_extractor(x: torch.Tensor) -> torch.Tensor:
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
    model = _get_pretrained()
    out = model(x)["out"]         # (B,21,H,W)
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

_FEATURE_EXTRACTORS: dict[str, nn.Module] = {}

@torch.no_grad()
def _extract_features_from_model(model: nn.Module, x: torch.Tensor, cache_key: str | None = None) -> torch.Tensor:
    """
    Extract intermediate feature maps from a pretrained model.
    
    Args:
        model: Pretrained PyTorch model
        x: Input tensor
        cache_key: Optional key to cache the feature extractor
    
    Returns:
        Feature map tensor (B,32,H,W)
    """
    xin = x
    x = _ensure_rgb_norm(x)
    
    # Check cache if key provided
    if cache_key and cache_key in _FEATURE_EXTRACTORS:
        feat = _FEATURE_EXTRACTORS[cache_key]
    else:
        # Build feature extractor (same logic as before)
        model_name = model.__class__.__name__.lower()
        
        if 'resnet' in model_name:
            feat = nn.Sequential(*list(model.children())[:-2])
        elif 'mobilenet' in model_name:
            feat = model.features
        elif 'efficientnet' in model_name:
            feat = model.features
        elif 'shufflenet' in model_name:
            feat = nn.Sequential(
                model.conv1, model.maxpool, model.stage2, # type: ignore
                model.stage3, model.stage4, # type: ignore
                model.conv5 if hasattr(model, 'conv5') else nn.Identity() # type: ignore
            )
        elif 'deeplabv3' in model_name:
            # DeepLabV3 backbone returns OrderedDict, wrap to extract the final feature map
            backbone = model.backbone
            def wrapped_backbone(x):
                feats = backbone(x) # type: ignore
                # feats is OrderedDict with keys like 'out', 'aux', or stage names
                # Return the highest-resolution feature map (usually 'out' or last entry)
                if isinstance(feats, dict):
                    if 'out' in feats:
                        return feats['out']
                    # Fallback: return last value in OrderedDict
                    return list(feats.values())[-1]
                return feats
            feat = wrapped_backbone
        else:
            feat = getattr(model, 'features', model)
        
        feat = feat.eval() if hasattr(feat, 'eval') else feat # type: ignore
        if hasattr(feat, 'parameters'):
            for p in feat.parameters(): # type: ignore
                p.requires_grad_(False)

        # Cache it
        if cache_key:
            _FEATURE_EXTRACTORS[cache_key] = feat # type: ignore
    
    # Execute feature extraction
    if callable(feat):
        y = feat(x)
    else:
        y = feat(x) # type: ignore
    
    # Handle OrderedDict output (shouldn't happen after wrapping, but safety check)
    if isinstance(y, dict):
        if 'out' in y:
            y = y['out']
        else:
            y = list(y.values())[-1]
    
    y = _project_channels_cached(y, 32) # type: ignore
    y = F.interpolate(y, size=_as_nchw(xin).shape[-2:], mode='bilinear', align_corners=False)
    
    return _restore_like(y, xin)


@torch.no_grad()
def _ensure_rgb_norm(x: torch.Tensor) -> torch.Tensor:
    """Ensure input tensor is 3-channel RGB and normalized."""
    x = _as_nchw(x).to(device)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        x = x[:, :3]
    return (x - _MEAN) / _STD

# ----------------- Custom model loader / wrapper -----------------
_CUSTOM_MODEL = None
_CUSTOM_MODEL_PATH = None
_CUSTOM_MODEL_BUILDER = None
_CUSTOM_MODEL_NORMALIZE = True

def set_custom_model(path: str, model_builder=None, normalize: bool = True, eager_load: bool = False):
    """
    Register a custom model checkpoint.
    path: TorchScript archive (.pt/.pth) or full pickled nn.Module or state_dict
    model_builder: constructor if checkpoint is a state_dict; None if full module / TorchScript
    normalize: apply ImageNet normalization before inference if True
    eager_load: load immediately (otherwise lazy on first inference)
    """
    global _CUSTOM_MODEL_PATH, _CUSTOM_MODEL_BUILDER, _CUSTOM_MODEL_NORMALIZE, _CUSTOM_MODEL
    _CUSTOM_MODEL_PATH = path
    _CUSTOM_MODEL_BUILDER = model_builder
    _CUSTOM_MODEL_NORMALIZE = normalize
    _CUSTOM_MODEL = None
    if eager_load:
        _load_custom_model()

def _load_custom_model():
    """Load and cache the custom model from disk.
    Returns: nn.Module"""
    global _CUSTOM_MODEL, _CUSTOM_MODEL_PATH, _CUSTOM_MODEL_BUILDER
    if _CUSTOM_MODEL is not None:
        return _CUSTOM_MODEL
    if _CUSTOM_MODEL_PATH is None:
        raise RuntimeError("Custom model path not set.")
    if not os.path.isfile(_CUSTOM_MODEL_PATH):
        raise FileNotFoundError(f"Custom model file not found: {_CUSTOM_MODEL_PATH}")

    # Try TorchScript first (silences the warning you saw)
    try:
        _CUSTOM_MODEL = torch.jit.load(_CUSTOM_MODEL_PATH, map_location=device)
        _CUSTOM_MODEL.eval()
        return _CUSTOM_MODEL
    except Exception:
        pass

    ckpt = torch.load(_CUSTOM_MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(ckpt, nn.Module):
        _CUSTOM_MODEL = ckpt.eval().to(device)
        return _CUSTOM_MODEL
    # state_dict or wrapped
    if _CUSTOM_MODEL_BUILDER is None:
        raise RuntimeError("state_dict detected but model_builder=None.")
    model = _CUSTOM_MODEL_BUILDER()
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    _CUSTOM_MODEL = model.eval().to(device)
    return _CUSTOM_MODEL

@torch.no_grad()
def custom_model_infer(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Run inference on the registered custom model.

    args:
        x: (H,W), (C,H,W) or (B,C,H,W) tensor
        scale: output scaling factor
    Returns: 
        output tensor (same batch shape as input)
    """
    m = _load_custom_model()
    xin = x
    x = _as_nchw(x)  # (B,C,H,W)
    B, C, H, W = x.shape

    # Infer required input channels from first 4D weight
    required_c = None
    for p in m.parameters():
        if p.dim() == 4:
            required_c = p.shape[1]
            break

    if required_c is not None and C != required_c:
        if C == 1 and required_c > 1:
            # replicate single channel
            x = x.repeat(1, required_c, 1, 1)
        elif C < required_c:
            # pad by repeating first channel
            pad_list = [x[:, :1]] * (required_c - C)
            x = torch.cat([x] + pad_list, dim=1)
        else:
            # truncate excess channels
            x = x[:, :required_c]

    # Normalization only if 3-channel ImageNet style
    if _CUSTOM_MODEL_NORMALIZE and x.shape[1] == 3:
        x = (x - _MEAN) / _STD

    y = m(x)

    if isinstance(y, dict):
        for v in y.values():
            if torch.is_tensor(v):
                y = v
                break
    if isinstance(y, (list, tuple)):
        y = y[0]

    y = _as_nchw(y) * float(scale) # type: ignore
    return _restore_like(y, xin)


# endregion

# region ==== Filtering & edge detection Functions ====

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



