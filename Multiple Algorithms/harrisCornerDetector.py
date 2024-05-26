import numpy as np
from scipy.ndimage import convolve
import cv2


def harrisCornerDetector(image, threshold=0.01, window_size=3, k=0.04, nms_window=3):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
    else:
        gray = image.astype(np.float32)

    # gray = cv2.GaussianBlur(gray, (3, 3), 2)
    # Compute gradients using Sobel filters
    Ix = convolve(gray, np.array([[-1, 0, 1]]), mode='constant')
    Iy = convolve(gray, np.array([[-1], [0], [1]]), mode='constant')

    # Compute elements of the structure tensor
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    # Compute the sums of the structure tensor elements over the window
    Sxx = convolve(Ixx, np.ones((window_size, window_size)), mode='constant')
    Syy = convolve(Iyy, np.ones((window_size, window_size)), mode='constant')
    Sxy = convolve(Ixy, np.ones((window_size, window_size)), mode='constant')

    # Compute the corner response function
    det = Sxx * Syy - Sxy**2
    trace = Sxx + Syy
    R = det - k * trace**2

    # Threshold the corner response
    corners = np.zeros_like(gray)
    corners[R > threshold * R.max()] = 255

    # Non-maximum suppression
    for y in range(0, corners.shape[0], nms_window):
        for x in range(0, corners.shape[1], nms_window):
            window = corners[y:y+nms_window, x:x+nms_window]
            if np.sum(window) > 255:
                max_pos = np.unravel_index(np.argmax(window), window.shape)
                corners[y:y+nms_window, x:x+nms_window] = 0
                corners[y+max_pos[0], x+max_pos[1]] = 255

    # Draw circles around the detected corners on the original image
    output_image = np.copy(image)
    corner_coords = np.argwhere(corners == 255)
    for y, x in corner_coords:
        cv2.circle(output_image, (x, y), 1, (0, 0, 255), 5)

    return output_image
