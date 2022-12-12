import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_on_best_channel(img: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on the `best` channel. """
    # YOUR CODE HERE: how do you define the best channel?
    # ...
    for i in range(3):
        plt.subplot(130+i+1)
        best_channel = img[:, :, i]
        plt.imshow(best_channel.astype('uint8') )
    plt.show()
    best_channel = img[:, :, 1]
    # or best_channel = img[:, :, 2]

    # YOUR CODE HERE: see cv2.Canny(...)
    out = cv2.Canny(best_channel, 100, 200, 3)
    return out

def process_on_intensity_channel(img: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on the intensity channel. """
    # YOUR CODE HERE: what is the best `intensity` channel?
    intensity = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,0]
    out = cv2.Canny(intensity, 100, 200, 3)
    return out

def parallel_channels_then_combine(img: np.ndarray) -> np.ndarray:
    """ Apply a Canny edge detector on each channel, then combine them. """
    # YOUR CODE HERE: see cv2.bitwise_or(...), cv2.bitwise_and(...)
    canny_per_channel = []
    for i in range(3):
        per_channel = img[:, :, i]
        canny_per_channel.append(cv2.Canny(per_channel, 100, 200, 3))
    out = np.zeros_like(img[:,:,0])
    for bin in canny_per_channel:
        out = cv2.bitwise_xor(out, bin)

    
    return out


if __name__ == "__main__":
    img_bgr = cv2.imread('../samples/peppers.tiff', cv2.IMREAD_COLOR)

    results = {
        # 'original': img_bgr,
        'Best channel': process_on_best_channel(img_bgr),
        'Intensity': process_on_intensity_channel(img_bgr),
        'In parallel then combine': parallel_channels_then_combine(img_bgr),
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
