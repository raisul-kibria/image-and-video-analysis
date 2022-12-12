import cv2
import numpy as np
from matplotlib import pyplot as plt


def hough_lines():
    """Plots the location of the main lines of the image."""
    # Your code here: see `cv.HoughLines(...)` and the sample `logo.png`
    img = cv2.imread('../samples/logo.png', cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200)
    theta = np.pi/180
    rhos = [1] #np.arange(1, 5)
    out = img.copy()
    for rho in rhos:
        line = cv2.HoughLines(img_edges, rho, theta, 128)
        if line is not None:
            for i in range(0, len(line) - 4):
                rho_h = line[i][0][0]
                theta_h = line[i][0][1]
                a = np.cos(theta_h)
                b = np.sin(theta_h)
                x0 = a * rho_h
                y0 = b * rho_h
                pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
                pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
                cv2.line(out, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    # Visualize them
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Hough (lines)')
    axs[0].set_title('Grayscale')
    axs[0].imshow(img_gray, cmap='gray')
    axs[1].set_title('Edges (dilated)')
    axs[1].imshow(cv2.dilate(img_edges, np.ones((5,5))), cmap='gray')
    axs[2].set_title('Img (with lines)')
    axs[2].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.show()


def hough_circles():
    """Plots the location of the main circles of the image."""
    # Your code here: see `cv.HoughCircles(...)` and the sample `dots.tiff`
    img = cv2.imread('../samples/dots.tiff', cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200)
    min_dist = 30
    out = img.copy()
    circle = cv2.HoughCircles(img_edges, cv2.HOUGH_GRADIENT, 1, min_dist, param1= 10, param2= 40)
    if circle.any():
        circles = np.uint16(np.around(circle))
        for i in circles[0,:]:
            out = cv2.circle(out,(i[0],i[1]),i[2], (0, 0, 255), 3)

    # Visualize them
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Hough (lines)')
    axs[0].set_title('Grayscale')
    axs[0].imshow(img_gray, cmap='gray')
    axs[1].set_title('Edges (dilated)')
    axs[1].imshow(cv2.dilate(img_edges, np.ones((5,5))), cmap='gray')
    axs[2].set_title('Img (with lines)')
    axs[2].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.show()


if __name__ == "__main__":
    hough_lines()
    hough_circles()
