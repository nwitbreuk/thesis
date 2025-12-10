import cv2
import numpy as np


def apply_to_mask(flag):
    def decorator(func):
        func.apply_to_mask = flag
        return func

    return decorator


@apply_to_mask(False)
def apply_bilateral_filter(
    batch_imgs, d=1, sigma_color=1, sigma_space=1, is_mask=False
):
    filtered_imgs = [
        cv2.bilateralFilter(img, int(d), sigma_color, sigma_space) for img in batch_imgs
    ]
    result = np.array(filtered_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(False)
def apply_histogram_equalization(batch_imgs, apply=1.0, is_mask=False):
    if round(apply):
        equalized_imgs = [cv2.equalizeHist(img) for img in batch_imgs]
        result = np.array(equalized_imgs)
    else:
        result = np.array(batch_imgs)

    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(False)
def apply_blur(batch_imgs, kernel_size=5, sigma=0, is_mask=False):
    blurred_imgs = [
        cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma) for img in batch_imgs
    ]
    result = np.array(blurred_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(False)
def apply_sharpen(batch_imgs, alpha=1.5, is_mask=False):
    # Base sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel = kernel * alpha
    sharpened_imgs = [cv2.filter2D(img, -1, kernel) for img in batch_imgs]
    result = np.array(sharpened_imgs)

    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)

    return result


@apply_to_mask(False)
def adjust_contrast(batch_imgs, alpha=1.0, is_mask=False):
    result = np.array([cv2.convertScaleAbs(img, alpha=alpha) for img in batch_imgs])
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(False)
def adjust_gamma(batch_imgs, gamma=1.0, is_mask=False):
    inv_gamma = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype(np.uint8)
    result = np.take(table, batch_imgs, axis=-1)
    return result


@apply_to_mask(True)
def rotate(batch_imgs, angle=90, is_mask=False):
    rotated_imgs = []
    for img in batch_imgs:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Choose interpolation method based on is_mask
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

        rotated_img = cv2.warpAffine(img, M, (w, h), flags=interp)
        rotated_imgs.append(rotated_img)
    result = np.array(rotated_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(True)
def translate(batch_imgs, translate_x=0, translate_y=0, is_mask=False):
    translated_imgs = []
    for img in batch_imgs:
        (h, w) = img.shape[:2]

        # Translation matrix
        M = np.float32([[1, 0, translate_x], [0, 1, translate_y]]) # type: ignore

        # Choose interpolation method based on is_mask
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

        translated_img = cv2.warpAffine(img, M, (w, h), flags=interp) # type: ignore
        translated_imgs.append(translated_img)
    result = np.array(translated_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(True)  # Change to False if you don't want this applied to masks
def resize_by_factor(batch_imgs, factor=1.0, is_mask=False):
    resized_imgs = []
    for img in batch_imgs:
        new_size = (int(img.shape[1] * factor), int(img.shape[0] * factor))

        # Choose interpolation method based on is_mask
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

        resized_img = cv2.resize(img, new_size, interpolation=interp)
        resized_imgs.append(resized_img)

    result = np.array(resized_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result


@apply_to_mask(False)  # Brightness adjustment typically doesn't apply to masks
def adjust_brightness(batch_imgs, delta=0, is_mask=False):
    adjusted_imgs = []
    for img in batch_imgs:
        # Ensure values wrap around in uint8 by using cv2.add
        adjusted_img = cv2.add(img, np.array([delta]))
        adjusted_imgs.append(adjusted_img)

    result = np.array(adjusted_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result

@apply_to_mask(False)
def solarize(batch_imgs, threshold, is_mask=False):
    adjusted_imgs = []
    for img in batch_imgs:
        # Ensure values wrap around in uint8 by using cv2.add
        adjusted_img = np.where(img>threshold, -img, img)
        adjusted_imgs.append(adjusted_img)
    
    result = np.array(adjusted_imgs)
    if len(result.shape) < 4:
        result = np.expand_dims(result, axis=-1)
    return result