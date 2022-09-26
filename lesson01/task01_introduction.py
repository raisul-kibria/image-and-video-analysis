import time
import numpy as np
import cv2
from matplotlib import pyplot as plt


# 1) Load the `mandril` image with OpenCV (see `cv2.imread(...)`)
img_path = "../samples/mandril.tiff"
img = cv2.imread(img_path)

# 2) Find dtype, size (in pixels), and number of channels of the image
print(
    f"Data type: {img.dtype}\nImage Shape: {img.shape[:-1]}\nChannels: {img.shape[-1]}"
)

# 3) Transform from BGR to RGB and Grayscale (see `cv2.cvtColor(...)`, `cv2.COLOR_BGR2RGB`, `cv2.COLOR_BGR2GRAY`)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4) Visualize with OpenCV (see `cv2.imshow(...)`)
cv2.imshow("Original Image", img)
cv2.imshow("RGB Image", img_rgb)
cv2.imshow("Grayscale Image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5) Visualize with Matplotlib (see `plt.imshow(...)`)
plt.figure()
plt.subplot(131)
plt.imshow(img[...,::-1])
plt.title("Original")
plt.subplot(132)
plt.imshow(img_rgb[...,::-1])
plt.title("RGB")
plt.subplot(133)
plt.imshow(img_gray[...,::-1])
plt.title("Grayscale")
plt.show()

# 6) Load video from file, and display it using OpenCV (see `cv2.VideoCapture(...)`)
vid_path =  "../samples/portitxol.mp4"
vid = cv2.VideoCapture(vid_path)
while True:
    _, frame = vid.read()
    cv2.imshow(vid_path.split("/")[-1].split(".")[0], frame)
    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()

# 7) If available, load video from webcam, and display it using OpenCV (use `cv2.VideoCapture(0)`)
vid = cv2.VideoCapture(0)
while True:
    _, frame = vid.read()
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()