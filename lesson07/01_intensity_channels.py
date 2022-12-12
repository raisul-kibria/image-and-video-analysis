import cv2
import numpy as np
from matplotlib import pyplot as plt


def intensity(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE
    return img_bgr[:,:,0] // 3 + img_bgr[:,:,1] // 3 + img_bgr[:,:,2] // 3


def luma(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE
    gray = img_bgr[:,:,0] * 0.0722+ img_bgr[:,:,1] * 0.7152 + img_bgr[:,:,2] * 0.2126
    return gray.astype('uint8')


def value(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:,:,-1]


def lightness_from_hsl(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)[:,:,1]


def lightness_from_cielab(img_bgr: np.ndarray) -> np.ndarray:
    """ Compute the intensity of an image. """
    # YOUR CODE HERE
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)[:,:,0]


if __name__ == "__main__":
    img_bgr = cv2.imread('../samples/lv.jpg', cv2.IMREAD_COLOR)

    results = {
        # 'original': img_bgr,
        'intensity': intensity(img_bgr),
        'luma': luma(img_bgr),
        'value': value(img_bgr),
        'L (HSL)': lightness_from_hsl(img_bgr),
        'L* (CIEL*a*b*)': lightness_from_cielab(img_bgr),
    }
    results = {k: v for k, v in results.items() if v is not None}

    # Visualize images
    fig, axs = plt.subplots(1, len(results))
    [ax.axis('off') for ax in axs.flatten()]
    # Show one image per subplot
    for ax, (title, subimage) in zip(axs.flatten(), results.items()):
        ax.imshow(cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    plt.show()
