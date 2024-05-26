import numpy as np
import cv2
import math
import numpy as np


def hough_line_transform_manual(image, rho_resolution=1, theta_resolution=1, threshold=50, num_lines=10):
    height, width = image.shape
    diag_len = int(math.sqrt(height ** 2 + width ** 2))
    rho_max = int(math.ceil(diag_len / rho_resolution))
    rhos = np.arange(-rho_max, rho_max + 1) * rho_resolution
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    edge_points = np.argwhere(image > 0)

    for point in edge_points:
        y, x = point
        for theta_index in range(len(thetas)):
            rho_float = x * cos_theta[theta_index] + y * sin_theta[theta_index]
            rho = int(np.round(rho_float / rho_resolution)) + rho_max
            accumulator[rho, theta_index] += 1

    lines = []
    for _ in range(num_lines):
        max_val = np.max(accumulator)
        if max_val <= threshold:
            break
        max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        rho = rhos[max_idx[0]]
        theta = thetas[max_idx[1]]
        lines.append((rho, theta))
        accumulator[max_idx] = 0

    return lines


def houghLineTransform(image, nLines, rho, theta):

    if nLines and rho:
        try:
            num_lines = int(nLines)
            srcImage = np.array(image)
            dst = cv2.Canny(srcImage, 50, 150, apertureSize=3)
            cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
            cdstP = np.copy(srcImage)

            # lines = cv2.HoughLinesP(dst, rho, np.pi / 180, 50, None, 50, 10)
            lines = hough_line_transform_manual(
                dst, rho, theta, 50, num_lines)
            if lines is not None:
                for rho, theta in lines:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(cdstP, (x1, y1), (x2, y2), (0, 0, 255), 2)
            return cdstP
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
