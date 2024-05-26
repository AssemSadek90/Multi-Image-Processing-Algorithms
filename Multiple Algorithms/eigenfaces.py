
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces


def eigenFaces(topK, imageNum):
    faces = fetch_olivetti_faces()
    images = faces.images
    
    X = np.array([img.flatten() for img in images])

    mean_face = np.mean(X, axis=0)

    X_centered = X - mean_face

    covariance_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    num_eigenfaces = topK
    eigenfaces = eigenvectors[:, :num_eigenfaces]

    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    variance_percentages = [0.10, 0.4, 0.7, 0.9]

    num_components = [np.argmax(cumulative_variance >= vp) +
                      1 for vp in variance_percentages]

    sample_image_index = imageNum  
    reconstructions = []

    for n in num_components:
        eigenfaces_n = eigenvectors[:, :n]
        weights = np.dot(X_centered[sample_image_index], eigenfaces_n)
        reconstruction = np.dot(weights, eigenfaces_n.T) + mean_face
        reconstructions.append(reconstruction)

    fig, axes = plt.subplots(7, 8, figsize=(15, 20))

    for i, ax in enumerate(axes[:5].flat):
        ax.imshow(eigenfaces[:, i].reshape(64, 64), cmap='gray')
        ax.axis('off')

    axes[5, 0].imshow(images[sample_image_index], cmap='gray')
    axes[5, 0].axis('off')
    axes[5, 0].text(0.5, -0.15, 'Original', ha='center',
                    va='top', transform=axes[5, 0].transAxes)

    for i, (reconstruction, n) in enumerate(zip(reconstructions, num_components)):
        ax = axes[5 + (i // 4), (i % 4) + 1]
        ax.imshow(reconstruction.reshape(64, 64), cmap='gray')
        ax.axis('off')
        ax.text(0.5, -0.15, f'{n} components', ha='center',
                va='top', transform=ax.transAxes)

    for j in range(len(reconstructions) + 1, 8):
        axes[5, j].axis('off')
    for j in range(8):
        axes[6, j].axis('off')

    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.tight_layout()
    plt.show()
