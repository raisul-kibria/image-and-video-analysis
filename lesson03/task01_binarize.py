from distutils.log import error
import cv2
import numpy as np
from matplotlib import pyplot as plt


def binarize_by_thresholding(img: np.ndarray, threshold: float) -> np.ndarray:
    """Returns a binary version of the image by applying a thresholding operation."""
    # YOUR CODE HERE
    out = img.copy()
    out[img > threshold] = 255
    out[img <= threshold] = 0
    return out

def binarize_by_otsu(img: np.ndarray) -> np.ndarray:
    """Returns a binary version of the image by applying a thresholding operation."""
    otsu_threshold = 0
    lowest_criteria = np.inf
    for threshold in range(255):
        lower_var = np.var(img[img < threshold])
        higher_var = np.var(img[img >= threshold])
        if np.sum([img < threshold] * 1) == 0 or np.sum([img >= threshold] * 1) == 0:
            continue

        metric = lower_var * np.sum([img < threshold] * 1)/img.size + higher_var * np.sum([img >= threshold] * 1)/img.size
        if metric < lowest_criteria:
            lowest_criteria = metric
            otsu_threshold = threshold
    return binarize_by_thresholding(img, otsu_threshold)


def binarize_by_dithering(img: np.ndarray) -> np.ndarray:
    """Returns a binary image by applying the Floyd–Steinberg dithering algorithm to a grayscale image."""
    # Add one extra row to avoid dealing with "corner cases" in the loop.
    padded_img = np.zeros(shape=(img.shape[0] + 1, img.shape[1] + 1), dtype=img.dtype)
    padded_img[:-1, :-1] = img
    threshold = 128
    weights = [7/16, 3/16, 5/16, 1/16]
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < threshold:
                err = threshold - img[i, j]
                # out[i, j] = 

            


if __name__ == "__main__":
    # Show effect
    original_img = cv2.imread('../samples/airplane.tiff', cv2.IMREAD_GRAYSCALE)  # Read the image.
    original_img = original_img.astype('float32')  # Convert to float32 to avoid overflow and rounding errors
    # img = cv2.resize(img, dsize=(img.shape[1]//20, img.shape[0]//20))

    for image in [original_img, original_img[175:225, 70:120]]:
        results = {
            'img': image,
            'Threshold_64': binarize_by_thresholding(image, 64),
            'Threshold_128': binarize_by_thresholding(image, 128),
            'Threshold_192': binarize_by_thresholding(image, 192),
            'Otsu': binarize_by_otsu(image),
            'Dithering': binarize_by_dithering(image),
        }
        results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

        # Visualize images
        fig, axs = plt.subplots(2, 3)
        # Remove default axis
        for ax in axs.flatten():
            ax.axis('off')
        # Show one image per subplot
        for ax, (title, subimage) in zip(axs.flatten(), results.items()):
            ax.set_title(title)
            ax.imshow(subimage, cmap='gray')
        # Display figure
        plt.show()
