import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_in_parallel(img: np.ndarray) -> np.ndarray:
    """ Apply a histogram equalization to all channels independently. """
    # YOUR CODE HERE: see cv2.dilate(...)
    # ...
    out_img = img.copy()
    clip_limit=5.0
    grid_size=(4, 4)
    cl = cv2.createCLAHE(clip_limit, grid_size)
    for i in range(3):
        out = cl.apply(img[:,:,i])
        out_img[:,:,i] = out
    return out_img

def process_intensity_channel_preserve_chroma(img: np.ndarray) -> np.ndarray:
    """ Apply a histogram equalization to intensity channel only. """
    # YOUR CODE HERE: what is the best intensity-and-chroma color space?
    # ...
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    intensity = img_lab[:,:,0]
    plt.imshow(intensity, cmap='gray')
    plt.show()
    clip_limit=5.0
    grid_size=(4, 4)
    cl = cv2.createCLAHE(clip_limit, grid_size)
    
    out = cl.apply(intensity)
    img_lab[:,:,0] = out
    plt.imshow(out, cmap='gray')
    plt.show()
    return cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)


if __name__ == "__main__":
    img_bgr = cv2.imread('../samples/peppers.tiff', cv2.IMREAD_COLOR)

    results = {
        # 'original': img_bgr,
        'In parallel then combine': process_in_parallel(img_bgr),
        'Intensity': process_intensity_channel_preserve_chroma(img_bgr),
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