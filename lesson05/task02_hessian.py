import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt, colors


def hessian_matrix(img_grayscale: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the hessian matrix of the image."""
    # Your code here: use `cv2.Sobel(..., dx= , dy= , ...)` to compute second-order derivatives
    img00 = cv2.Sobel(img_grayscale, cv2.CV_64F, dx = 2, dy = 0)
    img01 = cv2.Sobel(img_grayscale, cv2.CV_64F, dx = 1, dy = 1)
    img10 = cv2.Sobel(img_grayscale, cv2.CV_64F, dx = 1, dy = 1)
    img11 = cv2.Sobel(img_grayscale, cv2.CV_64F, dx = 0, dy = 2)
    return [img00, img01, img10, img11]


def hessian_eigenvalues(img_grayscale: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Returns the eigenvalues of the hessian matrix of the image."""
    # Your code here: remember that eigenvalues are solutions of `x^2 - trace * x + det = 0`
    H = hessian_matrix(img_grayscale)
    trace = H[0] + H[3]
    det = H[0] * H[3] - H[1] * H[2]
    x1 = (trace + np.sqrt(np.power(trace, 2) - 4 * det)) / 2
    x2 = (trace - np.sqrt(np.power(trace, 2) - 4 * det)) / 2
    return [x1, x2]


def cylinders(img_grayscale: np.ndarray) -> np.ndarray:
    """Returns the pixels of the image that correspond to dark cylinder-like structure."""
    # Your code here: remember that eigenvalues are solutions of `x^2 - trace * x + det = 0`
    # ...
    cylinders = np.zeros_like(img_grayscale)
    vmax, vmin = hessian_eigenvalues(img_grayscale)
    cylinders[vmax > 225] = 255
    cylinders[np.abs(vmin) > 20] = 255
    cylinders

    # Return value for visualization, whose pixels should be either 0 o 255.
    return cylinders


if __name__ == "__main__":
    img = cv2.imread('../samples/Retinal_DRIVE21_original.tif', cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Visualize `hessian_matrix(...)`
    hessian_dxdx, hessian_dxdy, hessian_dydx, hessian_dydy = hessian_matrix(img_gray)
    min_value = np.min([hessian_dxdx, hessian_dxdy, hessian_dydx, hessian_dydy])
    max_value = np.max([hessian_dxdx, hessian_dxdy, hessian_dydx, hessian_dydy])
    colormap_normalizer = colors.SymLogNorm(linthresh=0.1, vmin=min_value, vmax=max_value) #For better visualization
    _, axs = plt.subplots(2, 3)
    axs[0, 0].set_title('Original')
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Grayscale')
    axs[1, 0].imshow(img_gray, cmap='gray')
    axs[0, 1].set_title('hessian_dxdx')
    axs[0, 1].imshow(hessian_dxdx, cmap='gray', norm=colormap_normalizer)
    axs[0, 2].set_title('hessian_dxdy')
    axs[0, 2].imshow(hessian_dxdy, cmap='gray', norm=colormap_normalizer)
    axs[1, 1].set_title('hessian_dydx')
    axs[1, 1].imshow(hessian_dydx, cmap='gray', norm=colormap_normalizer)
    axs[1, 2].set_title('hessian_dydy')
    axs[1, 2].imshow(hessian_dydy, cmap='gray', norm=colormap_normalizer)
    plt.show()

    # Visualize `hessian_eigenvalues(...)`
    eigenvalue_1, eigenvalue_2 = hessian_eigenvalues(img_gray)
    hessian_det = eigenvalue_1 * eigenvalue_2
    hessian_trace = eigenvalue_1 + eigenvalue_2
    colormap_normalizer = colors.SymLogNorm(linthresh=0.1, vmin=np.min([eigenvalue_1, eigenvalue_2]), vmax=np.max([eigenvalue_1, eigenvalue_2])) # For better visualization
    _, axs = plt.subplots(2, 3)
    axs[0, 0].set_title('Original')
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Grayscale')
    axs[1, 0].imshow(img_gray, cmap='gray')
    axs[0, 1].set_title('eigenvalue_1')
    axs[0, 1].imshow(eigenvalue_1, cmap='gray', norm=colormap_normalizer)
    axs[0, 2].set_title('eigenvalue_2')
    axs[0, 2].imshow(eigenvalue_2, cmap='gray', norm=colormap_normalizer)
    axs[1, 1].set_title('det')
    axs[1, 1].imshow(hessian_det, cmap='gray', norm=colors.SymLogNorm(linthresh=0.1))
    axs[1, 2].set_title('trace')
    axs[1, 2].imshow(hessian_trace, cmap='gray', norm=colors.SymLogNorm(linthresh=0.1))
    plt.show()

    # Visualize `find_cylinders(...)`
    img_cylinders = cylinders(img_gray)
    img_with_cylinders = img.copy()
    img_with_cylinders[img_cylinders == 255, :] = [0, 255, 0]
    _, axs = plt.subplots(1, 2)
    # Show one image per subplot
    axs[0].set_title('Original')
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Cylinders')
    axs[1].imshow(cv2.cvtColor(img_with_cylinders, cv2.COLOR_BGR2RGB))
    plt.show()

