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

def _build_nadia_model(model_name):
    # 1. Define the correct architecture (DeepLabV3+, not just MobileNet backbone)
    model = deeplabv3_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        num_classes=21,
        aux_loss=True
    )
    # 2. Load weights handling the ['model'] key
    config = MODEL_CONFIGS[model_name]
    path = config.get("path", "")
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
    "aoi_1": {
        "path": "/dataB1/aoi/benchmarks/model_library/ensemble_models/v2/ensemble_model_1.pth",
        "builder": None,
        "feature_extractor": False,
        "is_segmenter": True,
        "is_custom": True,
        "prim_name": "aoi_1",
        "description": "Custom AOI ensemble model 1",
    },
    "aoi_2": {
        "path": "/dataB1/aoi/benchmarks/model_library/ensemble_models/v2/ensemble_model_2.pth",
        "builder": None,
        "feature_extractor": False,
        "is_segmenter": True,
        "is_custom": True,
        "prim_name": "aoi_2",
        "description": "Custom AOI ensemble model 2",
    },
    "aoi_3": {
        "path": "/dataB1/aoi/benchmarks/model_library/ensemble_models/v2/ensemble_model_3.pth",
        "builder": None,
        "feature_extractor": False,
        "is_segmenter": True,
        "is_custom": True,
        "prim_name": "aoi_3",
        "description": "Custom AOI ensemble model 2",
    },
    "aoi_4": {
        "path": "/dataB1/aoi/benchmarks/model_library/general_models/v2/ddrnet_0_6_57_2750.h5",
        "builder": None,
        "feature_extractor": False,
        "is_segmenter": True,
        "is_custom": True,
        "prim_name": "aoi_4",
        "description": "Custom AOI ensemble model 4",
    },
    "model1": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_12345.pth",
        "builder": lambda: _build_nadia_model("model1"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model1",      # optional, defaults to model_name
        "description": "Nadia's custom segmentation model 1",
    },
    "model2": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_32823.pth",
        "builder": lambda: _build_nadia_model("model2"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model2",
        "description": "Nadia's custom segmentation model 2",
    },
    "model3": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_57923.pth",
        "builder": lambda: _build_nadia_model("model3"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model3",
        "description": "Nadia's custom segmentation model 3",
    },
    "model4": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_70852.pth",
        "builder": lambda: _build_nadia_model("model4"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model4",
        "description": "Nadia's custom segmentation model 4",
    },
    "model5": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_97245.pth",
        "builder": lambda: _build_nadia_model("model5"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model5",
        "description": "Nadia's custom segmentation model 5",
    },
    "model6": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_126181.pth",
        "builder": lambda: _build_nadia_model("model6"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model6",
        "description": "Nadia's custom segmentation model 6",
    },
    "model7": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_10new.pth",
        "builder": lambda: _build_nadia_model("model7"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model7",
        "description": "Nadia's custom segmentation model 7",
    },
    "model8": {
        # Path is required by seggp_functions, but our builder handles the actual loading.
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_1000new.pth",
        "builder": lambda: _build_nadia_model("model8"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model8",
        "description": "Nadia's custom segmentation model 8",
    },
    "model9": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_9000new.pth",
        "builder": lambda: _build_nadia_model("model9"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model9",
        "description": "Nadia's custom segmentation model 9",
    },
    "model10": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_12000new.pth",
        "builder": lambda: _build_nadia_model("model10"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model10",
        "description": "Nadia's custom segmentation model 10",
    },
    "model11": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_15000new.pth",
        "builder": lambda: _build_nadia_model("model11"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model11",
        "description": "Nadia's custom segmentation model 11",
    },
    "model12": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_19000new.pth",
        "builder": lambda: _build_nadia_model("model12"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model12",
        "description": "Nadia's custom segmentation model 12",
    },
    "model13": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_22000new.pth",
        "builder": lambda: _build_nadia_model("model13"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model13",
        "description": "Nadia's custom segmentation model 13",
    },
    "model14": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_25000new.pth",
        "builder": lambda: _build_nadia_model("model14"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model14",
        "description": "Nadia's custom segmentation model 14",
    },
    "model15": {
        "path": "/dataB2/archive/home/nadia_dobreva/PyTorch_CIFAR10/pascal_models/state_dicts/pascalvoc_mobilenetv3_28000new.pth",
        "builder": lambda: _build_nadia_model("model11"),
        "feature_extractor": True,
        "is_segmenter": True,
        "is_custom": True,          # <---
        "prim_name": "model15",
        "description": "Nadia's custom segmentation model 15",
    },
    
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
     Return a callable(x) -> logits (B, k, H, W) for baseline evaluation.
    
     Args:
          model_name: Model identifier from MODEL_CONFIGS
          num_classes: Number of output classes (k)
          selected_classes: Optional list of class IDs to select (e.g., [15, 8, 12] for VOC)
    
     Behavior:
     1. VOC/Nadia models (21 output channels):
         - Without selection: returns all 21 channels (or projects if k != 21)
         - With selection [15,8,12]: slices channels 15,8,12 -> returns 3 channels [0,1,2]
       
     2. AOI models (5 output channels):
         - Already trained on remapped data [0,1,2,3,4]
         - Returns output as-is (no slicing needed)
       
     3. Feature extractors (backbones only):
         - Adds a random conv head to project features -> k channels
         - Baseline performance will be poor (random)
    """
    import seggp_functions as felgp_fs

    config = MODEL_CONFIGS[model_name]
    k = int(num_classes)
    is_segmenter = config.get("is_segmenter", False)

    # 1. Handle Segmentation Models (DeepLab, Nadia, Custom)
    if is_segmenter:
        if config.get("is_custom", False):
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

        # Wrapper to handle channel slicing/remapping
        def fn(x):
            """
            Infers segmentation logits and handles channel mapping.

            x: (B,C_in,H,W)
            returns: (B,k,H,W)
            """
            out = _raw_infer(x) # (B, C_out, H, W)
            out = felgp_fs._as_nchw(out)
            C_out = out.shape[1]
            
            # Case 1: AOI models already output k channels [0,1,2,3,4] correctly
            # No remapping needed - they're pre-trained on remapped data
            if "aoi" in model_name.lower():
                # AOI models output exactly k channels in correct order
                if C_out == k:
                    return out
                else:
                    # Unexpected: project to correct size
                    print(f"[Warning] AOI model {model_name} output {C_out} channels, expected {k}")
                    return felgp_fs._project_channels_cached(out, k)
            
            # Case 2: VOC/Nadia models (21 classes) - slice selected channels
            # E.g., if selected_classes=[15,8,12], extract channels [15,8,12] -> [0,1,2]
            if selected_classes and C_out > k:
                # Slice the specific channels corresponding to selected channels
                valid_indices = [c for c in selected_classes if c < C_out]
                if len(valid_indices) == k:
                    return out[:, valid_indices, :, :]
            
            # Case 3: No selection or size already matches
            if C_out == k:
                return out
            
            # Case 4: Fallback - project channels
            return felgp_fs._project_channels_cached(out, k)
            
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
        if config.get("is_custom", False):
            felgp_fs.set_custom_model(
                path=config["path"],
                model_builder=config.get("builder"),
                normalize=True,
                eager_load=False
            )
            prim_name = config.get("prim_name", model_name)
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