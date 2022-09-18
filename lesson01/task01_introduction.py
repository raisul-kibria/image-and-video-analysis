import time

import cv2
from matplotlib import pyplot as plt


# 1) Load the `mandril` image with OpenCV (see `cv2.imread(...)`)
img = cv2.imread('../samples/mandril.tiff')


# 2) Find dtype, size (in pixels), and number of channels of the image
if len(img.shape) == 3:
    print(f'Loaded image: dtype={img.dtype}, size={img.shape[0]} x {img.shape[1]} px, channels: {img.shape[2] }.')
else:
    print(f'Loaded image: dtype={img.dtype}, size={img.shape[0]} x {img.shape[1]} px.')


# 3) Transform from BGR to RGB and Grayscale (see `cv2.cvtColor(...)`, `cv2.COLOR_BGR2RGB`, `cv2.COLOR_BGR2GRAY`)
img_rgb = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB)
img_grayscale = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)


# 4) Visualize with OpenCV (see `cv2.imshow(...)`)
cv2.imshow("Title of the window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 5) Visualize with Matplotlib (see `plt.imshow(...)`)
fig, axs = plt.subplots(2, 2)
for ax in axs.flatten():    # Remove default axis
    ax.axis('off')
axs[0, 0].imshow(img), axs[0, 0].set_title('Loaded by CV (BGR)')
axs[0, 1].imshow(img_rgb), axs[0, 1].set_title('Converted to RGB')
axs[1, 0].imshow(img_grayscale, cmap='gray'), axs[1, 0].set_title('Converted to Grayscale')
plt.show()  # Display figure


# 6) Load video from file, and display it using OpenCV (see `cv2.VideoCapture(...)`)
cap = cv2.VideoCapture('../samples/portitxol.mp4')
while cap.isOpened():
    successful, frame = cap.read()          # Capture frame-by-frame
    if successful:
        cv2.imshow('Video viewer', frame)   # Display the frame
        cv2.waitKey(25)                     # Wait 25 ms
    else:
        cap.release()                       # Release the video capture object
cv2.destroyAllWindows()     # Closes window


# 7) If available, load video from webcam, and display it using OpenCV (use `cv2.VideoCapture(0)`)
cap = cv2.VideoCapture(0)
end_at = time.monotonic() + 10
while cap.isOpened():
    successful, frame = cap.read()          # Capture frame-by-frame
    if not successful or time.monotonic() > end_at:
        cap.release()                       # Release the video capture object
    else:
        cv2.imshow('Video viewer', frame)   # Display the frame
        cv2.waitKey(25)                     # Wait 25 ms
cv2.destroyAllWindows()
