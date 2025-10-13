import torch
import torch.nn.functional as F   # <-- this defines F
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # <-- this defines device
import sift_features
import numpy
from scipy import ndimage
from skimage.filters import gabor
import skimage
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

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
# region ==== Feature Extraction Functions ====
# def conVector(img):
#     """
#     Concatenate image arrays into a vector.
#     Args:
#         img: Image or list of arrays.
#     Returns:
#         Flattened vector.
#     """
#     try:
#         img_vector=numpy.concatenate((img))
#     except:
#         img_vector=img
#     return img_vector

# def histLBP(image, radius, n_points):
#     """
#     Compute the histogram of Local Binary Pattern (LBP) features for an image.
#     Args:
#         image: Input image.
#         radius: Radius for LBP.
#         n_points: Number of points for LBP.
#     Returns:
#         LBP histogram.
#     """
#     lbp = local_binary_pattern(image, n_points, radius, method='nri_uniform')
#     n_bins = 59
#     hist, ax = numpy.histogram(lbp, n_bins, (0, 59))
#     return hist

# def all_lbp(image):
#     """
#     Compute LBP histograms for all images in a batch.
#     Args:
#         image: Batch of images.
#     Returns:
#         Array of LBP histograms.
#     """
#     feature = []
#     for i in range(image.shape[0]):
#         feature_vector = histLBP(image[i,:,:], radius=1.5, n_points=8)
#         feature.append(feature_vector)
#     return numpy.asarray(feature)

# def HoGFeatures(image):
#     """
#     Compute Histogram of Oriented Gradients (HoG) features for an image.
#     Args:
#         image: Input image.
#     Returns:
#         HoG feature image or original image if computation fails.
#     """
#     try:
#         img, realImage = hog(image, orientations=9, pixels_per_cell=(8, 8),
#                     cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
#                     transform_sqrt=False, feature_vector=True)
#         return realImage
#     except:
#         return image

# def hog_features_patches(image, patch_size, moving_size):
#     """
#     Compute HoG features for patches in an image.
#     Args:
#         image: Input image.
#         patch_size: Size of each patch.
#         moving_size: Step size for moving window.
#     Returns:
#         Array of HoG features for patches.
#     """
#     img = numpy.asarray(image)
#     width, height = img.shape
#     w = int(width / moving_size)
#     h = int(height / moving_size)
#     patch = []
#     for i in range(0, w):
#         for j in range(0, h):
#             patch.append([moving_size * i, moving_size * j])
#     hog_features = numpy.zeros((len(patch)))
#     realImage = HoGFeatures(img)
#     for i in range(len(patch)):
#         hog_features[i] = numpy.mean(
#             realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
#     return hog_features

# def global_hog_small(image):
#     """
#     Compute HoG features for all images in a batch using small patches.
#     Args:
#         image: Batch of images.
#     Returns:
#         Array of HoG features.
#     """
#     feature = []
#     for i in range(image.shape[0]):
#         feature_vector = hog_features_patches(image[i,:,:], 4, 4)
#         feature.append(feature_vector)
#     return numpy.asarray(feature)

# def all_sift(image):
#     """
#     Compute SIFT features for all images in a batch.
#     Args:
#         image: Batch of images.
#     Returns:
#         Array of SIFT feature vectors.
#     """
#     width, height = image[0, :, :].shape
#     min_length = numpy.min((width, height))
#     feature = []
#     for i in range(image.shape[0]):
#         img = numpy.asarray(image[i, 0:width, 0:height])
#         extractor = sift_features.SingleSiftExtractor(min_length)
#         feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
#         w, h = feaArrSingle.shape
#         feature_vector = numpy.reshape(feaArrSingle, (h,))
#         feature.append(feature_vector)
#     return numpy.asarray(feature)
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
