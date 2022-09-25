import unittest
import cv2
import numpy as np
from matplotlib import pyplot as plt
import task02_histogram as task02


def negative(image: np.ndarray) -> np.ndarray:
    """Returns the negative of a grayscale image in [0, 255]."""
    # YOUR CODE HERE: You can assume image is of dtype float
    neg_image = 255.0 - image
    return neg_image


def log_transform(image: np.ndarray) -> np.ndarray:
    """Returns the log transformation of a grayscale image."""
    # YOUR CODE HERE
    log_image = (255 / np.log(256)) * np.log(1 + image)
    return log_image


def exp_transform(image: np.ndarray) -> np.ndarray:
    """Returns the exp transformation of a grayscale image, which should invert the log transformation."""
    # YOUR CODE HERE
    exp_image = np.exp(image/(255 / np.log(256))) - 1
    return exp_image


def gamma_transform(image: np.ndarray, gamma: float) -> np.ndarray:
    """Returns the gamma transformation of a grayscale image."""
    # YOUR CODE HERE
    return (255.0 / np.power(255.0, gamma)) * np.power(image, gamma)


def windowing(image: np.ndarray, lower_threshold: float, upper_threshold: float) -> np.ndarray:
    """Linear normalization assigning values lower or equal to lower_threshold to 0, and values greater or equal to upper_threshold to 255."""
    # YOUR CODE HERE
    wn_image = image.copy()
    wn_image[image <= lower_threshold] = 0
    wn_image[image >= upper_threshold] = 255
    return wn_image


def minmax_normalization(image: np.ndarray) -> np.ndarray:
    """Linear normalization assigning the lowest value to 0 and the highest value to 255."""
    # YOUR CODE HERE
    mm_image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255.0
    return mm_image


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Histogram equalization."""
    # YOUR CODE HERE
    hist = task02.histogram_count_values(image, 256)
    hist = hist / np.sum(hist)
    cm_hist = np.zeros(256)
    for k in range(0, 256):
        if k == 0:
            cm_hist[k] = hist[k]
        else:
            cm_hist[k] = hist[k] + cm_hist[k-1]
    cm_hist = minmax_normalization(cm_hist)
    image = cm_hist[image.astype('uint8')]
    return image


def clahe(image: np.ndarray, clip_limit=5.0, grid_size=(4, 4)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization."""
    image = image.astype('uint8')   # Ensure that the image is of type uint8
    # YOUR CODE HERE: Look up in the opencv documentation the function cv2.createCLAHE
    cl = cv2.createCLAHE(clip_limit, grid_size)
    out = cl.apply(image)
    return out



if __name__ == '__main__':
    # Execute tests for task03.
    test = unittest.main(module='test', defaultTest='TestLesson01Task03', exit=False)

    # Show effect on moon image.
    img = cv2.imread('../samples/moon.jpg', cv2.IMREAD_GRAYSCALE)  # Read the image.
    img = img.astype('float32')  # Convert to float32 to avoid overflow and rounding errors
    results = {
        'img': img,
        'negative': negative(img),
        'log_transform': log_transform(img),
        'exp_transform': exp_transform(img),
        'gamma_transform_0.5': gamma_transform(img, 0.5),
        'gamma_transform_2': gamma_transform(img, 2),
        'windowing_10-100': windowing(img, 10, 100),
        'minmax_normalization': minmax_normalization(img),
        'histogram_equalization': histogram_equalization(img),
        'clahe': clahe(img.astype('uint8'))
    }
    results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

    # Visualize images
    fig, axs = plt.subplots(3, 4)
    # Remove default axis
    for ax in axs.flatten():
        ax.axis('off')
    # Show one image per subplot
    for ax, (title, img) in zip(axs.flatten(), results.items()):
        ax.set_title(title)
        ax.imshow(img, cmap='gray')
    # Display figure
    plt.show()

    # Show color curves
    x = np.linspace(0, 255, 1000)
    results = {
        'unchanged': x,
        'negative': negative(x),
        'log_transform': log_transform(x),
        'exp_transform': exp_transform(x),
        'gamma_transform_0.5': gamma_transform(x, 0.5),
        'gamma_transform_2': gamma_transform(x, 2),
        'windowing': windowing(x, 50, 200),
    }
    results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

    fig, ax = plt.subplots()
    for name, y in results.items():
        ax.plot(x, y, label=name)
    ax.legend()
    plt.show()