import unittest

import numpy as np
import cv2
from matplotlib import pyplot as plt


def histogram_find_cuts(nbins: int) -> np.ndarray:
    """Sequence of limits of each bin (e.g. [0.0, 85.0, 170.0, 255.0] for 3 bins)."""
    # YOUR CODE HERE
    # ...


def histogram_count_values(image: np.ndarray, nbins: int) -> np.ndarray:
    """Creates a histogram of a grayscale image."""
    # YOUR CODE HERE
    # ...


def histogram_plot(image: np.ndarray, nbins) -> None:
    """Plots a histogram of a grayscale image."""
    # YOUR CODE HERE: You can use a bar plot `plt.bar(...)`
    # ...


if __name__ == '__main__':
    # Execute tests for task02.
    test = unittest.main(module='test', defaultTest='TestLesson01Task02', exit=False)

    # Load the image
    image = cv2.imread('../samples/tank.tiff')
    plt.imshow(image)
    plt.show()

    # Create the histograms
    for n in [8, 16, 32]:
        cuts = histogram_find_cuts(nbins=n)
        print(f'# For {n} bins:')
        print(f'Histogram bins are separated by: {"-".join(f"{c:.1f}" for c in np.nditer(cuts))}.')

        values = histogram_count_values(image, nbins=n)
        for start, end, value in zip(cuts[:-1], cuts[1:], values):
            print(f'[{start:5.1f}, {end:5.1f}): {value}')

        histogram_plot(image, nbins=n)
