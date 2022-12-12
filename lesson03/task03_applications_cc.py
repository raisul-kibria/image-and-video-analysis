import typing as tp

import cv2
import numpy as np
import matplotlib.pyplot as plt

import lesson03.task01_binarize as task01


def binarize_by_hysteresis(img: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """Returns a binary version of the image by applying a hysteresis operation."""
    out = np.zeros_like(img)
    binary_img = task01.binarize_by_thresholding(img, low_threshold)
    _, label_img = cv2.connectedComponents(binary_img.astype('uint8'))
    labels = np.unique(label_img)
    for label in labels:
        # YOUR CODE HERE: see `np.any(...)` and `np.all(...)`.
        # ...
        pass

    return out


def object_area(binary_img: np.ndarray) -> np.ndarray:
    """Returns the area of one object, passed as a binary image that contains only one connected component."""
    # YOUR CODE HERE:
    # ...


def object_centroid(binary_img: np.ndarray) -> tp.Tuple[float, float]:
    """Returns the centroid of one object, passed as a binary image that contains only one connected component."""
    centroid_x = 0
    centroid_y = 0
    # We will iterate over all pixels (which is simple but very slow... it could be improved by vectorizing the operation)
    for x in range(binary_img.shape[0]):
        for y in range(binary_img.shape[1]):
            # YOUR CODE HERE
            # ...
            pass

    return centroid_x, centroid_y


def largest_object(binary_img: np.ndarray) -> np.ndarray:
    """Returns a binary image with only the largest connected component."""
    _, label_img = cv2.connectedComponents(binary_img.astype('uint8'))
    largest_object_label = None
    largest_object_pixels = 0
    for label in range(np.max(label_img)+1):
        # YOUR CODE HERE
        # ...
        pass


def most_centered_object(binary_img: np.ndarray) -> np.ndarray:
    """Returns a binary image with only the most centered connected component."""
    # YOUR CODE HERE: see `cv2.ConnectedComponents`
    # ...


if __name__ == "__main__":
    # Visualize Hysteresis for different upper thresholds
    original_img = cv2.imread('../samples/airplane.tiff', cv2.IMREAD_GRAYSCALE)  # Read the image.
    original_img = original_img.astype('float32')  # Convert to float32 to avoid overflow and rounding errors
    for image in [original_img, original_img[175:225, 70:120]]:
        fig, axs = plt.subplots(2, 2)
        # Remove default axis
        for ax in axs.flatten():
            ax.axis('off')
        # Show one image per subplot
        axs.flatten()[0].set_title('Orginal')
        axs.flatten()[0].imshow(image, cmap='gray')
        for ax, upper_th in zip(axs.flatten()[1:], [192, 208, 224]):
            ax.set_title(f'Hysteresis (128, {upper_th:d})')
            ax.imshow(
                binarize_by_hysteresis(image, low_threshold=128, high_threshold=upper_th),
                cmap='gray')
        # Display figure
        plt.show()

    # Visualize CC properties
    image = task01.binarize_by_otsu(cv2.imread("../samples/dots.tiff", cv2.IMREAD_GRAYSCALE))
    largest_object_img = largest_object(image)
    most_centered_object_img = most_centered_object(image)

    results = {
        'img': image,
        'largest_obj': largest_object(image),
        'most_centered_obj': most_centered_object(image),
    }
    results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

    # Visualize images
    fig, axs = plt.subplots(2, 2)
    # Remove default axis
    for ax in axs.flatten():
        ax.axis('off')
    # Show one image per subplot
    for ax, (title, subimage) in zip(axs.flatten(), results.items()):
        ax.set_title(title)
        ax.imshow(subimage, cmap='gray')
    # Display figure
    plt.show()
