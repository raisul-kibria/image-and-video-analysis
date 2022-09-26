import math
import unittest
import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np

def histogram_find_cuts(nbins: int) -> np.ndarray:
    """Sequence of limits of each bin (e.g. [0.0, 85.0, 170.0, 255.0] for 3 bins)."""
    # YOUR CODE HERE
    min_idx = 0.0
    max_idx = 255.0
    limit_top = 255.0 / (nbins)
    seq_array = []
    i = min_idx
    while True:
        seq_array.append(i)
        i = i + limit_top
        if i >= max_idx:
            seq_array.append(max_idx)
            break
    return np.array(seq_array)

def histogram_count_values(image: np.ndarray, nbins: int) -> np.ndarray:
    """Creates a histogram of a grayscale image."""
    # YOUR CODE HERE
    image_copy = image.copy()
    hist = np.zeros(nbins)
    image_copy = np.multiply(np.divide(image_copy, 255), nbins)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if len(image.shape) > 2:
                pix = math.floor(image_copy[i, j, 0])
            else:
                pix = math.floor(image_copy[i, j])
            if pix >= nbins:
                pix = nbins - 1
            hist[pix] += 1.0
    return hist

def histogram_plot(image: np.ndarray, nbins) -> None:
    """Plots a histogram of a grayscale image."""
    # YOUR CODE HERE: You can use a bar plot `plt.bar(...)`
    hist = histogram_count_values(image, nbins)
    plt.bar(range(1, nbins+1), hist)
    plt.show()

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
