import math
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt


def kernel_squared_mean_filter(size: Tuple[int, int]) -> np.ndarray:
    """Returns a kernel the of given size for the mean filter."""
    # YOUR CODE HERE: see `np.ones(...)`
    kernel = 1 / (size[0] * size[1]) * np.ones(size)
    return kernel



def kernel_gaussian_filter(size: Tuple[int, int], sigma: float) -> np.ndarray:
    """Returns a kernel of the given size for the Gaussian filter."""
    # YOUR CODE HERE: see `np.exp(...)`
    kernel = np.ones(size, dtype=np.float64)
    for i in range(size[0]):
        for j in range(size[1]):
            kernel[i, j] = np.exp((np.power(i - size[0]//2, 2) + np.power(j - size[1]//2, 2)) / (-2 * sigma * sigma))
    return kernel/np.sum(kernel)


def kernel_sharpening(kernel_smoothing: np.ndarray, alpha: float) -> np.ndarray:
    """Returns a kernel for sharpening the image."""
    # YOUR CODE HERE: see `np.zeros(...)` and `np.zeros_like(...)`
    identity = np.zeros_like(kernel_smoothing)
    identity[identity.shape[0]//2, identity.shape[1]//2] = 1
    kernel = identity - kernel_smoothing
    return identity + alpha * kernel



def kernel_horizontal_derivative() -> np.ndarray:
    """Returns a 3x1 kernel for the horizontal derivative using first order central difference coefficients. """
    # YOUR CODE HERE
    return 0.5 * np.array([[1, 0, -1]])


def kernel_vertical_derivative() -> np.ndarray:
    """Returns a 1x3 kernel for the vertical derivative using first order central difference coefficients. """
    # YOUR CODE HERE: see `np.transpose(...)`
    return 0.5 * np.transpose(np.array([[1, 0, -1]]))

def kernel_sobel_horizontal() -> np.ndarray:
    """Returns the sobel operator for horizontal derivatives. """
    # YOUR CODE HERE
    kernel = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]
    return np.array(kernel)/np.sum(np.abs(kernel))


def kernel_sobel_vertical() -> np.ndarray:
    """Returns the sobel operator for vertical derivatives. """
    # YOUR CODE HERE: see `np.transpose(...)`
    kernel = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]
    return np.transpose(np.array(kernel))/np.sum(np.abs(kernel))


def kernel_LoG_filter() -> np.ndarray:
    """Returns a 3x3 kernel for the Laplacian of Gaussian filter."""
    # YOUR CODE HERE
    return kernel_horizontal_derivative() * kernel_gaussian_filter([3, 3], 1)


if __name__ == '__main__':
    # Show effect on subset of boat image.
    img = cv2.imread('../samples/boat.tiff', cv2.IMREAD_GRAYSCALE)  # Read the image.
    img = img.astype('float32')     # Convert to float32 to avoid overflow and rounding errors
    img = img[150:200, 150:200]     # Select a small window

    kernels = {
        'kernel_squared_mean_filter': kernel_squared_mean_filter(size=(3, 3)),
        'kernel_gaussian_filter': kernel_gaussian_filter(size=(3, 3), sigma=10.0),
        'kernel_sharpening': kernel_sharpening(kernel_squared_mean_filter(size=(3, 3)), alpha=2),
        'kernel_horizontal_derivative': kernel_horizontal_derivative(),
        'kernel_vertical_derivative': kernel_vertical_derivative(),
        'kernel_sobel_horizontal': kernel_sobel_horizontal(),
        'kernel_sobel_vertical': kernel_sobel_vertical(),
        'kernel_LoG_filter': kernel_LoG_filter(),
    }
    kernels = {k: v for k, v in kernels.items() if v is not None}  # Remove None values.

    for name, kernel in kernels.items():
        output = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        # Visualize images
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title(name)
        im = axs[0].imshow(kernel, cmap='gray')
        plt.colorbar(im, ax=axs[0])
        axs[1].imshow(output, cmap='gray')  # ddpeth=-1 means same as input
        # Display figure
        plt.show()
