from telnetlib import theNULL
import cv2
import numpy as np
import matplotlib.pyplot as plt

import task01_binarize as task01


def dilation(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the dilation of the binary/grayscale image with the given structuring element."""
    # YOUR CODE HERE: see `cv2.dilate`
    out = cv2.dilate(img, structuring_element)
    return out


def erosion(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the erosion of the binary/grayscale image with the given structuring element."""
    out = cv2.erode(img, structuring_element)
    return out


def opening(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the opening of the binary/grayscale image with the given structuring element."""
    # YOUR CODE HERE: see `cv2.dilate`, `cv2.erode` and `np.flip`
    out = dilation(erosion(img, structuring_element), np.flip(structuring_element))
    return out


def closing(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the closing of the binary/grayscale image with the given structuring element."""
    # YOUR CODE HERE: see `cv2.dilate`, `cv2.erode` and `np.flip`
    out = erosion(dilation(img, structuring_element), np.flip(structuring_element))
    return out


def morphological_gradient(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the morphological gradient of the binary/grayscale image with the given structuring element."""
    # YOUR CODE HERE
    return img - erosion(img, structuring_element)


def morphological_skeleton(img: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Returns the morphological skeleton of the binary/grayscale image considering Lantuéjoul's method."""
    # YOUR CODE HERE
    skel = []
    temp = img.copy()
    while True:
        toph = temp - opening(temp, structuring_element)
        if np.sum(toph) == 0:
            break
        skel.append(toph)
        temp = erosion(temp, structuring_element)
    return np.sum(skel, axis=0, out= np.zeros_like(img))



if __name__ == "__main__":
    # Visualize morphological operations on binary and grayscale images
    binary_img = task01.binarize_by_otsu(cv2.imread("../samples/dots.tiff", cv2.IMREAD_GRAYSCALE))
    grayscale_img = cv2.imread("../samples/mandril.tiff", cv2.IMREAD_GRAYSCALE)

    se = np.ones((3, 3), dtype='uint8')    # 8-connectivity.

    for image in [binary_img, grayscale_img, grayscale_img[150:250, 150:250]]:
        results = {
            'erosion': erosion(image, se),
            'dilation': dilation(image, se),
            'original': image,
            'opening': opening(image, se),
            'closing': closing(image, se),
            'gradient': morphological_gradient(image, se),
            'skeleton': morphological_skeleton(image, se),
        }
        results = {k: v for k, v in results.items() if v is not None}  # Remove None values.

        # Visualize images
        fig, axs = plt.subplots(3, 3)
        # Remove default axis
        for ax in axs.flatten():
            ax.axis('off')
        # Show one image per subplot
        for ax, (title, subimage) in zip(axs.flatten(), results.items()):
            ax.set_title(title)
            ax.imshow(subimage, cmap='gray')
        # Display figure
        plt.show()
