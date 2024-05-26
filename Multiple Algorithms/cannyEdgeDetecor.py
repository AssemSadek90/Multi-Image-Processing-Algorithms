import numpy as np
import cv2


def cannyEdgeDetector(image, k, sigma, lowThreshold, highThreshold):
    # Convert the image to grayscale if it is RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Smoothing the image using a Gaussian filter
    smoothed_image = cv2.GaussianBlur(image, (k, k), sigma)

    # Step 2: Calculate gradients using Sobel operators
    gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    # Step 3: Non-maximum suppression
    suppressed_image = np.zeros_like(gradient_magnitude)
    suppressed_image = np.uint8(suppressed_image)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            if (0 <= gradient_direction[i, j] < 22.5) or (157.5 <= gradient_direction[i, j] <= 180):
                neighbor1 = gradient_magnitude[i, j+1]
                neighbor2 = gradient_magnitude[i, j-1]
            elif 22.5 <= gradient_direction[i, j] < 67.5:
                neighbor1 = gradient_magnitude[i+1, j-1]
                neighbor2 = gradient_magnitude[i-1, j+1]
            elif 67.5 <= gradient_direction[i, j] < 112.5:
                neighbor1 = gradient_magnitude[i+1, j]
                neighbor2 = gradient_magnitude[i-1, j]
            else:
                neighbor1 = gradient_magnitude[i-1, j-1]
                neighbor2 = gradient_magnitude[i+1, j+1]
            if gradient_magnitude[i, j] >= neighbor1 and gradient_magnitude[i, j] >= neighbor2:
                suppressed_image[i, j] = gradient_magnitude[i, j]

    # Step 4: Double thresholding
    high_threshold = highThreshold
    low_threshold = lowThreshold

    strong_edges = (suppressed_image > high_threshold)
    weak_edges = (suppressed_image >= low_threshold) & (
        suppressed_image <= high_threshold)

    # Step 5: Edge tracking by hysteresis
    edges = np.zeros_like(suppressed_image)
    edges[strong_edges] = 255
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    edges[i, j] = 255

    return edges
