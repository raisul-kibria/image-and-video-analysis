from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt


def median_filter(image: np.ndarray, filter_size: Tuple[int, int]) -> np.ndarray:
    """Returns an image after applying the median filter of the given size."""
    img_sz_x, img_sz_y = image.shape
    out_sz_x = img_sz_x - filter_size[0] + 1  # Why?
    out_sz_y = img_sz_y - filter_size[1] + 1  # Why?
    out = np.zeros(shape=(out_sz_x, out_sz_y), dtype=image.dtype)
    for i in range(out_sz_x):
        for j in range(out_sz_y):
            # YOUR CODE HERE: see `np.median(...)`
            out[i, j] = np.median(image[i: i + filter_size[0], j : j + filter_size[1]])
    return out


def max_pooling(image: np.ndarray, pool_size: Tuple[int, int]) -> np.ndarray:
    """Returns an image after applying the max pooling of the given size."""
    img_sz_x, img_sz_y = image.shape
    out_sz_x = img_sz_x//pool_size[0]   # Why?
    out_sz_y = img_sz_y//pool_size[1]   # Why?
    out = np.zeros(shape=(out_sz_x, out_sz_y), dtype=image.dtype)
    for i in range(out_sz_x):
        for j in range(out_sz_y):
            # YOUR CODE HERE: see `np.max(...)`
            out[i, j] = np.max(image[i * pool_size[0]: (i + 1) * pool_size[0], j * pool_size[1] : (j + 1) * pool_size[1]])
    return out



if __name__ == '__main__':
    # Show effect on subset of boat image.
    img = cv2.imread('../samples/boat.tiff', cv2.IMREAD_GRAYSCALE)  # Read the image.
    img = img.astype('float32')     # Convert to float32 to avoid overflow and rounding errors
    img = img[150:200, 150:200]     # Select a small window

    results = {
        'median': median_filter(img, (5, 5)),
        'max_pooling': max_pooling(img, (5, 5)),

    }
    results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

    # Visualize images
    fig, axs = plt.subplots(1, 2)
    # Remove default axis
    for ax in axs.flatten():
        ax.axis('off')
    # Show one image per subplot
    for ax, (title, img) in zip(axs.flatten(), results.items()):
        ax.set_title(title)
        ax.imshow(img, cmap='gray')
    # Display figure
    plt.show()
