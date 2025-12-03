"""
Centralized registry for neural network models used in GP primitives.
Add new models here without modifying segGP_main.py.
"""
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, resnet18, resnet50, efficientnet_b0
import seggp_functions as felgp_fs

# Model definitions
MODEL_CONFIGS = {
    "deeplabv3_resnet50": {
        "builder": lambda: deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT),
        "feature_extractor": True,
        "description": "DeepLabV3 with ResNet-50 backbone (default)"
    },
    "shufflenet_v2_x1_0": {
        "builder": lambda: torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True),
        "feature_extractor": True,
        "description": "ShuffleNetV2-x1.0 (lightweight)"
    },
    "resnet18": {
        "builder": lambda: resnet18(weights='DEFAULT'),
        "feature_extractor": True,
        "description": "ResNet-18 feature extractor"
    },
    "resnet50": {
        "builder": lambda: resnet50(weights='DEFAULT'),
        "feature_extractor": True,
        "description": "ResNet-50 feature extractor"
    },
    "mobilenet_v3_small": {
        "builder": lambda: mobilenet_v3_small(weights='DEFAULT'),
        "feature_extractor": True,
        "description": "MobileNetV3-Small (lightweight)"
    },
    "efficientnet_b0": {
        "builder": lambda: efficientnet_b0(weights='DEFAULT'),
        "feature_extractor": True,
        "description": "EfficientNet-B0"
    },
    "custom_aoi": {
        "path": "/dataB1/aoi/benchmarks/model_library/ensemble_models/v2/ensemble_model_1.pth",
        "builder": None,  # loaded directly
        "feature_extractor": False,
        "description": "Custom AOI ensemble model"
    },
    "nadia_model": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_1000new.pth",
        "builder": lambda: mobilenet_v3_large(num_classes=21),   # loaded directly
        "feature_extractor": True, 
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

def make_segmenter(model_name, num_classes: int):
    """
    Return a callable(x) -> logits (B, k, H, W) that runs the model as a segmenter.
    For feature-extractor backbones, we project features to k-class logits.
    For custom models (aoi/nadia), we use custom_model_infer and project if needed.
    """
    import seggp_functions as felgp_fs

    config = MODEL_CONFIGS[model_name]
    k = int(num_classes)

    if model_name in ["custom_aoi", "nadia_model"]:
        # Use custom model infra; load now for baseline
        felgp_fs.set_custom_model(
            path=config["path"],
            model_builder=config.get("builder"),  # None for both models
            normalize=True,
            eager_load=True
        )
        def fn(x):
            out = felgp_fs.custom_model_infer(x, scale=1.0)           # (B,C?,H,W)
            out4 = felgp_fs._as_nchw(out)
            if out4.shape[1] != k:
                return felgp_fs.feature_to_logits(out4, k)
            return out4
        return fn

    # Feature-extractor models: build extractor then project to k
    feat_extractor_fn = make_feat_extractor(model_name)
    def fn(x):
        feats = feat_extractor_fn(x)                                  # (B,32,H,W)
        return felgp_fs.feature_to_logits(feats, k)                   # (B,k,H,W)
    return fn

def register_model_primitives(pset, models_to_use, color_mode, run_mode):
    """
    Register model primitives based on configuration.
    
    Args:
        pset: DEAP primitive set
        models_to_use: List of model names from MODEL_CONFIGS
        color_mode: "rgb" or "gray"
        run_mode: "fast", "middle", "normal", or "aoi"
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
        if model_name in ["custom_aoi", "nadia_model"]:  # ✅ Add nadia_model here
            felgp_fs.set_custom_model(
                path=config["path"],
                model_builder=config.get("builder"),  # None for both models
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