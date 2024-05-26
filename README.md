# Image Processing Algorithms Application

This project is a comprehensive image processing application developed using PyQt, featuring multiple algorithms for edge detection, corner detection, line detection, ellipse detection, and face reconstruction. The application is structured into a tabbed interface, where each tab is dedicated to a specific algorithm.

## Features

1. **Tabbed Interface:**
   - Each tab represents a different image processing algorithm.
   - The available algorithms include:
     - Canny Edge Detection
     - Harris Corner Detection
     - Hough Line Detection
     - Hough Ellipse Detection
     - Eigenfaces Face Reconstruction using Principal Component Analysis (PCA)

2. **Parameter Input:**
   - Users can input algorithm-specific parameters within each tab to customize the processing.

3. **Image Browsing:**
   - Users can double-click on the image container in each tab to browse and select an input image.

4. **Output Display:**
   - Each tab contains two rows of three image containers:
     - The top row displays the input images.
     - The bottom row displays the corresponding output images after processing.

## Algorithms

### Canny Edge Detection
Allows users to adjust parameters such as the lower and upper thresholds for edge detection.

**Demo Image:**
![Canny Edge Detection]("Multiple Algorithms/images/Canny Demo.png")

### Harris Corner Detection
Users can modify parameters like the sensitivity threshold for detecting corners.

**Demo Image:**
![Harris Corner Detection]("Multiple Algorithms/images/Harris Corner Demo.png")

### Hough Line Detection
Parameters include rho resolution, theta resolution, and threshold for line detection.

**Demo Image:**
![Hough Line Detection](Multiple%20Algorithms/images/Hough%20Line%20Demo.png)

### Hough Ellipse Detection
Users can specify parameters such as the accuracy and minimum distance between detected ellipses.

**Demo Image:**
![Hough Ellipse Detection]("Multiple Algorithms/images/Hough Ellipse Demo.png")

### Eigenfaces Face Reconstruction
Users can input the number of principal components to be used for face reconstruction.

**Demo Image:**
![Eigenfaces Face Reconstruction]("Multiple Algorithms/images/Eigen Faces Demo.png")

## User Interaction

- Double-clicking the image container within a tab opens a file browser for selecting an image.
- Upon selection, the chosen image is displayed in the input container, and the processed result is shown in the output container below.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/image-processing-app.git
   cd image-processing-app
