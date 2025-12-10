"""
Centralized registry for neural network models used in GP primitives.
Add new models here without modifying segGP_main.py.
"""
import os
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, deeplabv3_mobilenet_v3_large
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, resnet18, resnet50, efficientnet_b0, MobileNet_V3_Large_Weights
import seggp_functions as felgp_fs

def _build_nadia_model():
    # 1. Define the correct architecture (DeepLabV3+, not just MobileNet backbone)
    model = deeplabv3_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        num_classes=21,
        aux_loss=True
    )
    # 2. Load weights handling the ['model'] key
    path = "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_1000new.pth"
    if os.path.exists(path):
        print(f"[Model Registry] Loading custom weights from {path}")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"[Warning] Nadia model path not found: {path}")
    return model

# Model definitions
MODEL_CONFIGS = {
    "deeplabv3_resnet50": {
        "builder": lambda: deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT),
        "feature_extractor": True,
        "is_segmenter": True,  # ✅ Mark as a full segmentation model
        "description": "DeepLabV3 with ResNet-50 backbone (default)"
    },
    "shufflenet_v2_x1_0": {
        "builder": lambda: torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', weights = 'DEFAULT'),
        "feature_extractor": True,
        "is_segmenter": False,
        "description": "ShuffleNetV2-x1.0 (lightweight)"
    },
    "resnet18": {
        "builder": lambda: resnet18(weights='DEFAULT'),
        "feature_extractor": True,
        "is_segmenter": False,
        "description": "ResNet-18 feature extractor"
    },
    "resnet50": {
        "builder": lambda: resnet50(weights='DEFAULT'),
        "feature_extractor": True,
        "is_segmenter": False,
        "description": "ResNet-50 feature extractor"
    },
    "mobilenet_v3_small": {
        "builder": lambda: mobilenet_v3_small(weights='DEFAULT'),
        "feature_extractor": True,
        "is_segmenter": False,
        "description": "MobileNetV3-Small (lightweight)"
    },
    "efficientnet_b0": {
        "builder": lambda: efficientnet_b0(weights='DEFAULT'),
        "feature_extractor": True,
        "is_segmenter": False,
        "description": "EfficientNet-B0"
    },
    "custom_aoi": {
        "path": "/dataB1/aoi/benchmarks/model_library/ensemble_models/v2/ensemble_model_1.pth",
        "builder": None,  # loaded directly
        "feature_extractor": False,
        "is_segmenter": True,
        "description": "Custom AOI ensemble model"
    },
    "nadia_model": {
        # Path is required by seggp_functions, but our builder handles the actual loading.
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_1000new.pth",
        "builder": _build_nadia_model,  # ✅ Use the custom builder defined above
        "feature_extractor": True, 
        "is_segmenter": True,
        "description": "Nadia's custom segmentation model"
    }
}

# Wrap model as feature extractor
def make_feat_extractor(model_name):
    _cached_model = None  # Closure variable to cache the model
    def feat_fn(x):
        nonlocal _cached_model
        if _cached_model is None:
            # Build model once and cache it
            _cached_model = MODEL_CONFIGS[model_name]["builder"]()
            _cached_model.eval()
            # Freeze parameters
            for p in _cached_model.parameters():
                p.requires_grad_(False)
        
        # Move model to the same device as input
        device = x.device if hasattr(x, 'device') else 'cpu'
        _cached_model = _cached_model.to(device)
        
        # Extract features using the cached model
        return felgp_fs._extract_features_from_model(_cached_model, x, cache_key=model_name)
    return feat_fn

def make_segmenter(model_name, num_classes: int, selected_classes: list[int] | None = None):
    """
    Return a callable(x) -> logits (B, k, H, W).
    
    If the model is a known segmenter (DeepLab, Custom), we attempt to slice
    the specific output channels corresponding to 'selected_classes'.
    Otherwise, we project features using a random head (baseline will be poor).
    """
    import seggp_functions as felgp_fs

    config = MODEL_CONFIGS[model_name]
    k = int(num_classes)
    is_segmenter = config.get("is_segmenter", False)

    # 1. Handle Segmentation Models (DeepLab, Nadia, Custom)
    if is_segmenter:
        # Load model logic
        if model_name in ["custom_aoi", "nadia_model"]:
            felgp_fs.set_custom_model(
                path=config["path"],
                model_builder=config.get("builder"),
                normalize=True,
                eager_load=True
            )
            def _raw_infer(x):
                return felgp_fs.custom_model_infer(x, scale=1.0)
        elif model_name == "deeplabv3_resnet50":
            # Use the specific pretrained function that returns logits
            def _raw_infer(x):
                return felgp_fs.pretrained_seg_nn(x)
        else:
            # Fallback for other segmenters if added
            feat_fn = make_feat_extractor(model_name)
            _raw_infer = feat_fn

        # Wrapper to handle channel slicing
        def fn(x):
            out = _raw_infer(x) # (B, C_out, H, W)
            out = felgp_fs._as_nchw(out)
            
            # If we have specific classes to select (e.g. [15, 8, 12])
            if selected_classes is not None and len(selected_classes) == k:
                # Filter out ignore index if present in selection (though usually handled by caller)
                valid_indices = [c for c in selected_classes if c != 255]
                
                # Check if indices are within model output range
                if valid_indices and max(valid_indices) < out.shape[1]:
                    # ✅ SLICE: Pick exactly the channels we want
                    return out[:, valid_indices, :, :]
            
            # Fallback: if shapes match exactly
            if out.shape[1] == k:
                return out
                
            # Fallback: Random projection (Performance will be low/random)
            return felgp_fs.feature_to_logits(out, k)
            
        return fn

    # 2. Handle Feature Extractors (ResNet, MobileNet backbones)
    # These are NOT segmentation models, so baseline performance is expected to be near zero/random
    # unless we trained a head (which we don't do in baseline mode).
    feat_extractor_fn = make_feat_extractor(model_name)
    def fn_backbone(x):
        feats = feat_extractor_fn(x)                                  # (B,32,H,W)
        return felgp_fs.feature_to_logits(feats, k)                   # (B,k,H,W)
    return fn_backbone

def register_model_primitives(pset, models_to_use, color_mode, run_mode):
    """
    Register model primitives based on configuration.
    """
    import seg_types
    tp = felgp_fs.trace_primitive
    
    _INPUT_TYPE = seg_types.RGBImage if color_mode == "rgb" else seg_types.GrayImage
    
    for model_name in models_to_use:
        if model_name not in MODEL_CONFIGS:
            print(f"[Warning] Unknown model: {model_name}, skipping")
            continue
        
        config = MODEL_CONFIGS[model_name]
        
        # Skip heavy models in fast mode
        if run_mode == "fast" and model_name in ["resnet50", "efficientnet_b0"]:
            continue
        
        # Register primitive based on model type
        if model_name in ["custom_aoi", "nadia_model"]:
            felgp_fs.set_custom_model(
                path=config["path"],
                model_builder=config.get("builder"),
                normalize=True,
                eager_load=False
            )
            prim_name = "aoiModel" if model_name == "custom_aoi" else "nadiaModel"
            pset.addPrimitive(
                tp(prim_name, felgp_fs.custom_model_infer),
                [_INPUT_TYPE, float],
                seg_types.FeatureMap,
                name=prim_name
            )
        elif config["feature_extractor"]:            
            pset.addPrimitive(
                tp(f"{model_name}_feat", make_feat_extractor(model_name)),
                [_INPUT_TYPE],
                seg_types.FeatureMap,
                name=f"{model_name}_feat"
            )
    
    print(f"[Model Registry] Registered {len(models_to_use)} models: {models_to_use}")