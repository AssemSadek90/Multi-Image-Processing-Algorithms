import cv2
import numpy as np
import cannyEdgeDetecor

# Load and preprocess the image


def houghEllipse(image, k, sigma, a_low, a_high, b_low, b_high):

    print("Image loaded.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Image converted to grayscale.")
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    print("Gaussian blur applied.")
    edges = cannyEdgeDetecor.cannyEdgeDetector(blurred, 7, sigma, 50, 150)
    print("Edge detection done.")

    # Parameters
    a_range = (a_low, a_high)
    b_range = (b_low, b_high)
    theta_range = np.deg2rad(np.arange(0, 90))  # Convert theta to radians
    print("Parameters set.")

    # Initialize the accumulator as a dictionary
    accumulator = {}
    print("Accumulator initialized.")

    # Voting
    for y, x in np.argwhere(edges):
        for a in range(*a_range):
            for b in range(*b_range):
                for theta in theta_range:
                    x_c = int(round(x - a * np.cos(theta)))
                    y_c = int(round(y + b * np.sin(theta)))
                    if 0 <= x_c < image.shape[1] and 0 <= y_c < image.shape[0]:
                        key = (y_c, x_c, a, b, theta)
                        if key in accumulator:
                            accumulator[key] += 1
                        else:
                            accumulator[key] = 1
    print("Voting completed.")

    # Find potential ellipses
    threshold = max(accumulator.values()) * 0.5
    potential_ellipses = [k for k, v in accumulator.items() if v >= threshold]
    print(f"Potential ellipses identified: {len(potential_ellipses)}")

    # Ellipse Fitting
    scale_factor = 0.1  # Further reduce the size of the ellipse
    for y_c, x_c, a, b, theta in potential_ellipses:
        scaled_a = int(a * scale_factor)
        scaled_b = int(b * scale_factor)
        cv2.ellipse(image, (x_c, y_c), (scaled_a, scaled_b),
                    np.degrees(theta), 0, 360, (0, 0, 255), 1)
    print("Ellipses drawn on image.")

    return (image)
