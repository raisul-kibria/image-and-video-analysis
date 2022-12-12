import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt


def gaussian_pyramid(img: np.ndarray, levels: int) -> tp.List[np.ndarray]:
    """Returns a Gaussian pyramid of the image."""
    # Your code here: see `cv.pyrDown(...)`
    img_array = []
    img_copy = img.copy()
    img_size = np.array(img.shape)
    for _ in range(levels):
        img_array.append(img_copy)
        img_size = np.divide(img_size, 2).astype('uint8')
        img_copy = cv2.pyrDown(img_copy, dstsize= img_size)
    return img_array


def laplacian_pyramid(img: np.ndarray, levels: int) -> tp.List[np.ndarray]:
    """Returns a Laplacian pyramid of the image."""
    # Your code here: see `cv.pyrDown(...)` and `cv.pyrUp(...)`
    gauss_dn = gaussian_pyramid(img, levels)
    out = [gauss_dn[-1]]
    for i in range(levels - 2, -1, -1):
        img_inp = gauss_dn[i + 1]
        target_shape = np.multiply(img_inp.shape, 2)
        recon = cv2.pyrUp(img_inp, dstsize = target_shape)
        out.append(gauss_dn[i] - recon)
    return out[::-1]



def reconstruct_from_laplacian_pyramid(l_pyramid: tp.List[np.ndarray]) -> np.ndarray:
    """Reconstructs an image from its Laplacian pyramid."""
    # Your code here: see `cv.pyrUp(...)`, and start from the smallest layer.
    last_recon = l_pyramid[-1]
    for i in range(len(l_pyramid) - 1, 0, -1):
        last_recon = cv2.pyrUp(last_recon) + l_pyramid[i - 1]
    return last_recon

def remove_finer_detail(img: np.ndarray) -> np.ndarray:
    """Removes the finer details of the image by applying a Laplacian pyramid."""
    original_img = cv2.imread('../samples/airplane.tiff', cv2.IMREAD_GRAYSCALE).astype('float32')
    l_pyramid = laplacian_pyramid(original_img, levels=6)
    # Your code here: how many `levels` will you remove? how?
    levels_removed = 1
    out = reconstruct_from_laplacian_pyramid(l_pyramid[levels_removed:6])
    for _ in range(levels_removed):
        out = cv2.pyrUp(out)
    return out


if __name__ == "__main__":
    # Show effect
    original_img = cv2.imread('../samples/airplane.tiff', cv2.IMREAD_GRAYSCALE)  # Read the image.
    original_img = original_img.astype('float32')  # Convert to float32 to avoid overflow and rounding errors

    pyramids = {
        'Gaussian': gaussian_pyramid(original_img, 4),
        'Laplacian': laplacian_pyramid(original_img, 4),
    }
    pyramids = {k: v for k, v in pyramids.items() if v is not None}  # Remove None values.

    for method_name, pyramid in pyramids.items():
        # Visualize images
        fig, axs = plt.subplots(1, 4)
        fig.suptitle(method_name)
        # Show one image per subplot
        for ax, subimage in zip(axs.flatten(), pyramid):
            ax.imshow(subimage, cmap='gray')
        # Display figure
        plt.show()

    # Show effect of removing coarse details
    if 'Laplacian' in pyramids:
        l_pyramid = pyramids['Laplacian']
        reconstructions = {
            'Original': original_img,
            'Recovered': reconstruct_from_laplacian_pyramid(l_pyramid),
            'Without coarse detail': remove_finer_detail(original_img),
        }
        reconstructions = {k: v for k, v in reconstructions.items() if v is not None}  # Remove None values.
        # Visualize images
        fig, axs = plt.subplots(1, 3)
        # Show one image per subplot
        for ax, (name, subimage) in zip(axs.flatten(), reconstructions.items()):
            ax.set_title(name)
            ax.imshow(subimage, cmap='gray')
        # Display figure
        plt.show()
