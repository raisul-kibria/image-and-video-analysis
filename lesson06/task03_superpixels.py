import cv2
import numpy as np
from matplotlib import pyplot as plt


def superpixels_LSC(img_bgr: np.ndarray) -> np.ndarray:
    """ Returns a label image corresponding to the superpixels created according to the LSC algorithm. """
    # Your code here: see class `cv2.ximgproc.createSuperpixelLSC`, and its methods `iterate(...)` and `getLabels(...)`
    # ...
    lscObj = cv2.ximgproc.createSuperpixelLSC(img_bgr, 40)
    lscObj.iterate(15)
    return lscObj.getLabels()


def superpixels_SEEDS(img_bgr: np.ndarray) -> np.ndarray:
    """ Returns a label image corresponding to the superpixels created according to the SEEDS algorithm. """
    # Your code here: see class `cv2.ximgproc.createSuperpixelSEEDS`, and its methods `iterate(...)` and `getLabels(...)`
    seedsObj = cv2.ximgproc.createSuperpixelSEEDS(img_bgr.shape[0], img_bgr.shape[1], img_bgr.shape[2], 40, 5)
    seedsObj.iterate(img_bgr, 15)
    return seedsObj.getLabels()


def superpixels_SLIC(img_bgr: np.ndarray) -> np.ndarray:
    """ Returns a label image corresponding to the superpixels created according to the SLIC algorithm. """
    # Your code here: see class `cv2.ximgproc.createSuperpixelSLIC`, and its methods `iterate(...)` and `getLabels(...)`
    seeds = cv2.ximgproc.createSuperpixelSLIC(img_bgr, region_size=30, ruler=3)
    seeds.iterate(num_iterations=5)
    return seeds.getLabels()


if __name__ == "__main__":
    img_bgr = cv2.imread('../samples/Ki67.jpg', cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.25, fy=0.25)

    results = {
        'LSC': superpixels_LSC(img_bgr),
        'SEEDS': superpixels_SEEDS(img_bgr),
        'SLIC': superpixels_SLIC(img_bgr),
    }
    results = {k: v for k, v in results.items() if v is not None}

    # Visualize images
    fig, axs = plt.subplots(2, 3)
    axs = axs.flatten()
    [ax.axis('off') for ax in axs]
    # Show one image per subplot
    for idx, (title, region_labels) in enumerate(results.items()):
        # Plot borders
        subimage_borders = np.copy(img_bgr)
        subimage_borders[region_labels != cv2.erode(region_labels.astype('uint8'), np.ones((3, 3))), ...] = (0, 255, 0)
        axs[idx].imshow(cv2.cvtColor(subimage_borders, cv2.COLOR_BGR2RGB))
        # Plot average color
        subimage_avrg_color = np.zeros_like(img_bgr)
        for idx_l in range(np.max(region_labels) + 1):
            subimage_avrg_color[region_labels == idx_l, ...] = np.mean(img_bgr[region_labels == idx_l, ...], axis=0)
        axs[3+idx].imshow(cv2.cvtColor(subimage_avrg_color, cv2.COLOR_BGR2RGB))
        axs[3+idx].set_title(title)
    plt.show()
