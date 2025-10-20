import torch
import torch.nn.functional as F   # <-- this defines F
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # <-- this defines device
import torchvision.models.segmentation as seg_models
import torch.nn as nn

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
    """
    Apply Gaussian blur to a tensor with a specified sigma.
    Args:
        x: Input tensor of shape (B, C, H, W).
        sigma: Standard deviation for Gaussian kernel.
    Returns:
        Blurred tensor.
    """
    # sigma in [0.5, 3]; auto kernel size
    sigma = float(max(0.5, min(3.0, sigma)))
    k = int(max(3, 2 * int(3 * sigma) + 1))
    ax = torch.arange(-k // 2 + 1., k // 2 + 1., device=x.device)
    g1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1 = g1 / g1.sum()
    k2 = torch.outer(g1, g1).unsqueeze(0).unsqueeze(0)
    return F.conv2d(x, k2, padding=k // 2)

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
_pretrained_seg_model = seg_models.deeplabv3_resnet50(pretrained=True).to(device)
_pretrained_seg_model.eval()
for param in _pretrained_seg_model.parameters():
    param.requires_grad = False  # Freeze the model

def pretrained_seg_nn(x):
    """
    Apply pre-trained segmentation NN to input tensor x (C,H,W).
    Assumes x is RGB (3 channels); outputs a binary segmentation map (1,H,W) for "horse" class.
    """
    if x.shape[0] != 3:
        # If not RGB, replicate grayscale to 3 channels (hack for grayscale inputs)
        x = x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3]  # Take first 3 if more
    
    with torch.no_grad():
        # Normalize to ImageNet stats (required for pre-trained models)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
        x_norm = (x - mean) / std
        
        # Forward pass
        out = _pretrained_seg_model(x_norm.unsqueeze(0))['out']  # Add batch dim; output shape (1,21,H,W)
        
        # Extract horse class (class 18 in COCO) and sigmoid for probability
        horse_prob = torch.sigmoid(out[:, 18:19])  # (1,1,H,W)
        return horse_prob.squeeze(0)  # (1,H,W)




# endregion

# region ==== Feature Extraction Functions ====

# endregion

# region ==== Filtering & edge detection Functions ====

# Sobel and Laplacian filters
sobel_x_kernel = torch.tensor([[[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]], dtype=torch.float32, device=device).unsqueeze(0)
sobel_y_kernel = torch.tensor([[[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]]], dtype=torch.float32, device=device).unsqueeze(0)
laplacian_kernel = torch.tensor([[[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]]], dtype=torch.float32, device=device).unsqueeze(0)

def conv2d(x, kernel):
    return F.conv2d(x, kernel, padding=1)

def sobel_x(x): return conv2d(x, sobel_x_kernel)
def sobel_y(x): return conv2d(x, sobel_y_kernel)
def laplacian(x): return conv2d(x, laplacian_kernel)

def gradient_magnitude(x):
    gx, gy = sobel_x(x), sobel_y(x)
    return torch.sqrt(gx ** 2 + gy ** 2)

def erode(x, k=3):
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k//2)

def dilate(x, k=3):
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k//2)

def open_f(x, k=3):
    return dilate(erode(x, k), k)

def close_f(x, k=3):
    return erode(dilate(x, k), k)


def gaussian_blur(x, k=5, sigma=1.0):
    def get_gaussian_kernel(k, sigma):
        ax = torch.arange(-k // 2 + 1., k // 2 + 1.)
        kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel
    gk = get_gaussian_kernel(k, sigma).to(device)
    kernel2d = torch.outer(gk, gk).unsqueeze(0).unsqueeze(0)
    return F.conv2d(x, kernel2d, padding=k//2)

def mean_filter(x, k=3):
    kernel = torch.ones((1, 1, k, k), device=device) / (k * k)
    return F.conv2d(x, kernel, padding=k//2)


# endregion
