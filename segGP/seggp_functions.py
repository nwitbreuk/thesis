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
def combine(*args):
    """
    Combine multiple arrays by summing them element-wise.
    Args:
        *args: Arrays to combine.
    Returns:
        Array: Element-wise sum of all input arrays.
    """
    output = args[0]
    for i in range(1, len(args)):
        output += args[i]
    return output
def FeaCon2(img1, img2):
    """
    Concatenate features from two images for each sample.
    Args:
        img1: First image array.
        img2: Second image array.
    Returns:
        Array of concatenated feature vectors.
    """
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        feature_vector = numpy.concatenate((image1, image2), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def FeaCon3(img1, img2, img3):
    """
    Concatenate features from three images for each sample.
    Args:
        img1: First image array.
        img2: Second image array.
        img3: Third image array.
    Returns:
        Array of concatenated feature vectors.
    """
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def FeaCon4(img1, img2, img3, img4):
    """
    Concatenate features from four images for each sample.
    Args:
        img1: First image array.
        img2: Second image array.
        img3: Third image array.
        img4: Fourth image array.
    Returns:
        Array of concatenated feature vectors.
    """
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        image4 = conVector(img4[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3, image4), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)
# endregion
# region ==== Classifier Functions ====
def linear_svm(x_train, y_train, cm=0):
    """
    Train a linear SVM classifier and return predicted labels.
    Args:
        x_train: Training data.
        y_train: Training labels.
        cm: Exponent for regularization parameter C.
    Returns:
        Predicted labels or probabilities.
    """
    c = 10**(cm)
    classifier = LinearSVC(C=c)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = svm_train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function_svm(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def lr(x_train, y_train, cm=0):
    """
    Train a logistic regression classifier and return predicted labels.
    Args:
        x_train: Training data.
        y_train: Training labels.
        cm: Exponent for regularization parameter C.
    Returns:
        Predicted labels or probabilities.
    """
    c = 10**(cm)
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = LogisticRegression(C=c, solver='sag', max_iter=1000)
    num_train = y_train.shape[0]
    if num_train==x_train.shape[0]:
        y_labels = svm_train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function_svm(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def randomforest(x_train, y_train, n_tree = 500, max_dep = 100):
    """
    Train a random forest classifier and return predicted probabilities.
    Args:
        x_train: Training data.
        y_train: Training labels.
        n_tree: Number of trees.
        max_dep: Maximum depth of trees.
    Returns:
        Predicted probabilities.
    """
    classifier = RandomForestClassifier(n_estimators=n_tree, max_depth=max_dep)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = train_model_prob(classifier, x_train, y_train)
    else:
        y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def erandomforest(x_train, y_train, n_tree = 500, max_dep = 100):
    """
    Train an extra trees classifier and return predicted probabilities.
    Args:
        x_train: Training data.
        y_train: Training labels.
        n_tree: Number of trees.
        max_dep: Maximum depth of trees.
    Returns:
        Predicted probabilities.
    """
    classifier = ExtraTreesClassifier(n_estimators=n_tree, max_depth=max_dep)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = train_model_prob(classifier, x_train, y_train)
    else:
        y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def svm_train_model(model, x, y, k=3):
    """
    Train a model using stratified k-fold cross-validation and return predicted labels.
    Args:
        model: Classifier model.
        x: Input data.
        y: Labels.
        k: Number of folds.
    Returns:
        Predicted labels (one-hot encoded).
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(np.asarray(x))
    kf = StratifiedKFold(n_splits=k)
    ni = np.unique(y)
    num_class = ni.shape[0]
    y_predict = np.zeros((len(y), num_class))
    for train_index, test_index in kf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        y_label = []
        for i in y_pred:
            binary_label = np.zeros((num_class))
            binary_label[int(i)] = 1
            y_label.append(binary_label)
        y_predict[test_index,:] = np.asarray(y_label)
    return y_predict

def test_function_svm(model, x_train, y_train, x_test):
    """
    Train a model and predict labels for test data.
    Args:
        model: Classifier model.
        x_train: Training data.
        y_train: Training labels.
        x_test: Test data.
    Returns:
        Predicted labels (one-hot encoded).
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(np.asarray(x_train))
    x_test = min_max_scaler.transform(np.asarray(x_test))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_label = []
    ni = np.unique(y_train)
    num_class = ni.shape[0]
    for i in y_pred:
        binary_label = np.zeros((num_class))
        binary_label[int(i)] = 1
        y_label.append(binary_label)
    y_predict = np.asarray(y_label)
    return y_predict

def train_model_prob(model, x, y, k=3):
    """
    Train a model using stratified k-fold cross-validation and return predicted probabilities.
    Args:
        model: Classifier model.
        x: Input data.
        y: Labels.
        k: Number of folds.
    Returns:
        Predicted probabilities.
    """
    kf = StratifiedKFold(n_splits=k)
    ni = np.unique(y)
    num_class = ni.shape[0]
    y_predict = np.zeros((len(y), num_class))
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train,y_train)
        y_predict[test_index,:] = model.predict_proba(x_test)
    return y_predict

def test_function_prob(model, x_train, y_train, x_test):
    """
    Train a model and predict probabilities for test data.
    Args:
        model: Classifier model.
        x_train: Training data.
        y_train: Training labels.
        x_test: Test data.
    Returns:
        Predicted probabilities.
    """
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_test)
    return y_pred
# endregion
# region ==== Feature Extraction Functions ====
def conVector(img):
    """
    Concatenate image arrays into a vector.
    Args:
        img: Image or list of arrays.
    Returns:
        Flattened vector.
    """
    try:
        img_vector=numpy.concatenate((img))
    except:
        img_vector=img
    return img_vector

def histLBP(image, radius, n_points):
    """
    Compute the histogram of Local Binary Pattern (LBP) features for an image.
    Args:
        image: Input image.
        radius: Radius for LBP.
        n_points: Number of points for LBP.
    Returns:
        LBP histogram.
    """
    lbp = local_binary_pattern(image, n_points, radius, method='nri_uniform')
    n_bins = 59
    hist, ax = numpy.histogram(lbp, n_bins, (0, 59))
    return hist

def all_lbp(image):
    """
    Compute LBP histograms for all images in a batch.
    Args:
        image: Batch of images.
    Returns:
        Array of LBP histograms.
    """
    feature = []
    for i in range(image.shape[0]):
        feature_vector = histLBP(image[i,:,:], radius=1.5, n_points=8)
        feature.append(feature_vector)
    return numpy.asarray(feature)

def HoGFeatures(image):
    """
    Compute Histogram of Oriented Gradients (HoG) features for an image.
    Args:
        image: Input image.
    Returns:
        HoG feature image or original image if computation fails.
    """
    try:
        img, realImage = hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                    transform_sqrt=False, feature_vector=True)
        return realImage
    except:
        return image

def hog_features_patches(image, patch_size, moving_size):
    """
    Compute HoG features for patches in an image.
    Args:
        image: Input image.
        patch_size: Size of each patch.
        moving_size: Step size for moving window.
    Returns:
        Array of HoG features for patches.
    """
    img = numpy.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
    hog_features = numpy.zeros((len(patch)))
    realImage = HoGFeatures(img)
    for i in range(len(patch)):
        hog_features[i] = numpy.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
    return hog_features

def global_hog_small(image):
    """
    Compute HoG features for all images in a batch using small patches.
    Args:
        image: Batch of images.
    Returns:
        Array of HoG features.
    """
    feature = []
    for i in range(image.shape[0]):
        feature_vector = hog_features_patches(image[i,:,:], 4, 4)
        feature.append(feature_vector)
    return numpy.asarray(feature)

def all_sift(image):
    """
    Compute SIFT features for all images in a batch.
    Args:
        image: Batch of images.
    Returns:
        Array of SIFT feature vectors.
    """
    width, height = image[0, :, :].shape
    min_length = numpy.min((width, height))
    feature = []
    for i in range(image.shape[0]):
        img = numpy.asarray(image[i, 0:width, 0:height])
        extractor = sift_features.SingleSiftExtractor(min_length)
        feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
        w, h = feaArrSingle.shape
        feature_vector = numpy.reshape(feaArrSingle, (h,))
        feature.append(feature_vector)
    return numpy.asarray(feature)
# endregion
# region ==== Filtering & edge detection Functions ====

def gau(left, si):
    """
    Apply a Gaussian filter to each image in a batch.
    Args:
        left: Batch of images.
        si: Sigma for Gaussian filter.
    Returns:
        Batch of filtered images.
    """
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_filter(left[i, :, :], sigma=si))
    return np.asarray(img)

def gauD(left, si, order):
    """
    Apply a Gaussian derivative filter to each image in a batch.
    Args:
        left: Batch of images.
        si: Sigma for Gaussian filter.
        order: Derivative order.
    Returns:
        Batch of filtered images.
    """
    img  = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_filter(left[i,:,:], sigma=si, order=order))
    return np.asarray(img)

def gab(left, the, fre):
    """
    Apply a Gabor filter to each image in a batch.
    Args:
        left: Batch of images.
        the: Theta (orientation).
        fre: Frequency.
    Returns:
        Batch of filtered images.
    """
    fmax = numpy.pi / 2
    a = numpy.sqrt(2)
    freq = fmax / (a ** fre)
    thea = numpy.pi * the / 8
    img = []
    for i in range(left.shape[0]):
        filt_real, filt_imag = numpy.asarray(gabor(left[i,:,:], theta=thea, frequency=freq))
        img.append(filt_real)
    return numpy.asarray(img)

def gaussian_Laplace1(left):
    """
    Apply a Gaussian Laplace filter (sigma=1) to an image or batch.
    Args:
        left: Image or batch of images.
    Returns:
        Filtered image or batch.
    """
    return ndimage.gaussian_laplace(left, sigma=1)

def gaussian_Laplace2(left):
    """
    Apply a Gaussian Laplace filter (sigma=2) to an image or batch.
    Args:
        left: Image or batch of images.
    Returns:
        Filtered image or batch.
    """
    return ndimage.gaussian_laplace(left, sigma=2)

def laplace(left):
    """
    Apply a Laplace filter to an image or batch.
    Args:
        left: Image or batch of images.
    Returns:
        Filtered image or batch.
    """
    return ndimage.laplace(left)

def sobelxy(left):
    """
    Apply a Sobel filter (both axes) to each image in a batch.
    Args:
        left: Batch of images.
    Returns:
        Batch of filtered images.
    """
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i, :, :]))
    return np.asarray(img)

def sobelx(left):
    """
    Apply a Sobel filter (x-axis) to each image in a batch.
    Args:
        left: Batch of images.
    Returns:
        Batch of filtered images.
    """
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i,:,:], axis=0))
    return np.asarray(img)

def sobely(left):
    """
    Apply a Sobel filter (y-axis) to each image in a batch.
    Args:
        left: Batch of images.
    Returns:
        Batch of filtered images.
    """
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i, :, :], axis=1))
    return np.asarray(img)

def maxf(image):
    """
    Apply a maximum filter to each image in a batch.
    Args:
        image: Batch of images.
    Returns:
        Batch of filtered images.
    """
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.maximum_filter(image[i,:,:], size))
    return np.asarray(img)

def medianf(image):
    """
    Apply a median filter to each image in a batch.
    Args:
        image: Batch of images.
    Returns:
        Batch of filtered images.
    """
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.median_filter(image[i,:,:], size))
    return np.asarray(img)

def meanf(image):
    """
    Apply a mean filter to each image in a batch.
    Args:
        image: Batch of images.
    Returns:
        Batch of filtered images.
    """
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.convolve(image[i,:,:], numpy.full((3, 3), 1 / (size * size))))
    return np.asarray(img)

def minf(image):
    """
    Apply a minimum filter to each image in a batch.
    Args:
        image: Batch of images.
    Returns:
        Batch of filtered images.
    """
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.minimum_filter(image[i,:,:], size))
    return np.asarray(img)

def lbp(image):
    """
    Compute Local Binary Pattern (LBP) for each image in a batch.
    Args:
        image: Batch of images.
    Returns:
        Batch of LBP images (normalized).
    """
    img = []
    for i in range(image.shape[0]):
        lbp = local_binary_pattern(image[i,:,:], 8, 1.5, method='nri_uniform')
        img.append(np.divide(lbp, 59))
    return np.asarray(img)

def hog_feature(image):
    """
    Compute HoG feature images for a batch of images.
    Args:
        image: Batch of images.
    Returns:
        Batch of HoG feature images.
    """
    try:
        img = []
        for i in range(image.shape[0]):
            img1, realImage = hog(image[i, :, :], orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                                transform_sqrt=False, feature_vector=True)
            img.append(realImage)
        data = np.asarray(img)
    except:
        data = image
    return data

def mis_match(img1, img2):
    """
    Crop two image batches to the same minimum width and height.
    Args:
        img1: First image batch.
        img2: Second image batch.
    Returns:
        Tuple of cropped image batches.
    """
    n, w1, h1 = img1.shape
    n, w2, h2 = img2.shape
    w = min(w1, w2)
    h = min(h1, h2)
    return img1[:, 0:w, 0:h], img2[:, 0:w, 0:h]

def mixconadd(img1, w1, img2, w2):
    """
    Weighted addition of two image batches after cropping to same size.
    Args:
        img1: First image batch.
        w1: Weight for first image batch.
        img2: Second image batch.
        w2: Weight for second image batch.
    Returns:
        Weighted sum of image batches.
    """
    img11, img22 = mis_match(img1, img2)
    return numpy.add(img11 * w1, img22 * w2)

def mixconsub(img1, w1, img2, w2):
    """
    Weighted subtraction of two image batches after cropping to same size.
    Args:
        img1: First image batch.
        w1: Weight for first image batch.
        img2: Second image batch.
        w2: Weight for second image batch.
    Returns:
        Weighted difference of image batches.
    """
    img11, img22 = mis_match(img1, img2)
    return numpy.subtract(img11 * w1, img22 * w2)

def sqrt(left):
    """
    Compute the square root of an array, replacing NaN/inf with 1.
    Args:
        left: Input array.
    Returns:
        Array with square roots.
    """
    with numpy.errstate(divide='ignore', invalid='ignore'):
        x = numpy.sqrt(left,)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1
            x[numpy.isnan(x)] = 1
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1
    return x

def relu(left):
    """
    Apply ReLU activation (max(0, x)) to an array.
    Args:
        left: Input array.
    Returns:
        Array after ReLU.
    """
    return (abs(left) + left) / 2

def maxP(left, kel1, kel2):
    """
    Apply block-wise max pooling to each image in a batch.
    Args:
        left: Batch of images.
        kel1: Pooling window height.
        kel2: Pooling window width.
    Returns:
        Batch of pooled images.
    """
    img = []
    for i in range(left.shape[0]):
        current = skimage.measure.block_reduce(left[i,:,:], (kel1, kel2),numpy.max) # type: ignore
        img.append(current)
    return np.asarray(img)
# endregion
