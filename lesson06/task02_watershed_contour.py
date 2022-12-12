import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt


def segmentation_by_watershed(img_bgr: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering a watershed method."""
    # Your code here: Smooth image to improve results (see `cv2.GaussianBlur(...)`)
    # ...
    img_bgr = cv2.GaussianBlur(img_bgr, (5,5), 0)

    # Your code here: Initialize markers (mark as 0 unknown pixels, as 1 background, as 2 foreground)
    markers = np.zeros(img_bgr.shape[:2], dtype=np.int32)
    markers = markers + 1
    rad = 80
    x_s = seed_pixel[0]-rad if seed_pixel[0]-rad > 0 else 0
    x_e = seed_pixel[0]+rad if seed_pixel[0]+rad < img_bgr.shape[0] else img_bgr.shape[0]

    y_s = seed_pixel[1]-rad if seed_pixel[1]-rad > 0 else 0
    y_e = seed_pixel[1]+rad if seed_pixel[1]+rad < img_bgr.shape[1] else img_bgr.shape[1]

    markers[x_s:x_e, y_s:y_e] = 0
    markers[seed_pixel] = 2
    plt.imshow(markers)
    plt.show()
    # Your code here: Apply watershed transformation (see `cv2.watershed(...)`)
    markers = cv2.watershed(img_bgr, markers)


    return markers


def contour_based_segmentation(img_gray: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering contours derived from edges."""
    # Your code here: see cv2.Canny(...) and cv2.findContours(..., mode=cv2.RETR_EXTERNAL, ...).
    # ...
    edge = cv2.Canny(img_gray, 100, 200)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Select only the contour containing the seed point
    for contour in contours:
        region_mask = np.zeros_like(img_gray)
        # Your code here: Draw the contour and its interior on the mask (see `cv2.drawContours(...)`)
        # ...
        region_mask = cv2.drawContours(region_mask, [contour], 0, 255, -1)
        # region_mask = cv2.dilate(region_mask, kernel=np.ones((3, 3)))
        if region_mask[seed_pixel]:
            return region_mask

    return np.zeros_like(img_gray)


if __name__ == "__main__":
    img_bgr = cv2.imread('../samples/Ki67.jpg', cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.25, fy=0.25)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Detect region of interest (where brown cells might should be present)
    region_of_interest = cv2.erode((img_gray <= 128).astype('uint8')*255, np.ones((3, 3)))
    # Randomly select a seed within the region of interest
    positive_points = np.where(region_of_interest != 0)
    seed_idx = np.random.choice(len(positive_points[0]))
    seed_point = positive_points[0][seed_idx], positive_points[1][seed_idx]

    results = {
        'Region of interest': region_of_interest,
        'Segmentation by watershed': segmentation_by_watershed(img_bgr, seed_point),
        'Edge-based segmentation': contour_based_segmentation(img_gray, seed_point),
    }
    results = {k: v for k, v in results.items() if v is not None}

    # Visualize images
    fig, axs = plt.subplots(2, 2)
    [ax.axis('off') for ax in axs.flatten()]
    # Show one image per subplot
    for ax, (title, binary_image) in zip(axs.flatten(), results.items()):
        subimage = np.copy(img_bgr)
        subimage[binary_image != cv2.erode(binary_image.astype('uint8'), np.ones((3, 3))), ...] = (0, 255, 0)
        cv2.line(subimage, pt1=(seed_point[1] - 5, seed_point[0] - 5), pt2=(seed_point[1] + 5, seed_point[0] + 5),
                 color=(0, 0, 255), thickness=2)
        cv2.line(subimage, pt1=(seed_point[1] - 5, seed_point[0] + 5), pt2=(seed_point[1] + 5, seed_point[0] - 5),
                 color=(0, 0, 255), thickness=2)
        ax.imshow(cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    plt.show()
