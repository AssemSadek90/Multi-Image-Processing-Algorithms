import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

def eigenFacesNew(topK, imageNum):
    faces = fetch_olivetti_faces()
    images = faces.images

    individual_index = imageNum  
    num_images_per_individual = 10  

    start_index = individual_index * num_images_per_individual
    end_index = start_index + num_images_per_individual
    individual_images = images[start_index:end_index]

    X_individual = np.array([img.flatten() for img in individual_images])

    mean_face_individual = np.mean(X_individual, axis=0)

    X_centered_individual = X_individual - mean_face_individual

    covariance_matrix_individual = np.cov(X_centered_individual, rowvar=False)

    eigenvalues_individual, eigenvectors_individual = np.linalg.eigh(covariance_matrix_individual)

    sorted_indices_individual = np.argsort(eigenvalues_individual)[::-1]
    eigenvalues_individual = eigenvalues_individual[sorted_indices_individual]
    eigenvectors_individual = eigenvectors_individual[:, sorted_indices_individual]

    num_eigenfaces_individual = topK 
    eigenfaces_individual = eigenvectors_individual[:, :num_eigenfaces_individual]

    sample_image_index = imageNum  
    reconstructions = []

    variance_percentages = [0.10, 0.15, 0.30, 0.60]

    cumulative_variance = np.cumsum(eigenvalues_individual) / np.sum(eigenvalues_individual)
    num_components_list = sorted(set(np.argmax(cumulative_variance >= vp) + 1 for vp in variance_percentages))

    for n in num_components_list:
        eigenfaces_n = eigenvectors_individual[:, :n]
        weights = np.dot(X_centered_individual[sample_image_index], eigenfaces_n)
        reconstruction = np.dot(weights, eigenfaces_n.T) + mean_face_individual
        reconstructions.append(reconstruction)

    fig, axes = plt.subplots(2, max(num_eigenfaces_individual, len(num_components_list) + 1), figsize=(15, 5))

    for i, ax in enumerate(axes[0, :num_eigenfaces_individual]):
        ax.imshow(eigenfaces_individual[:, i].reshape(64, 64), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Eigenface {i+1}')

    axes[1, 0].imshow(individual_images[sample_image_index], cmap='gray')
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Original')

    for i, (reconstruction, n) in enumerate(zip(reconstructions, num_components_list)):
        ax = axes[1, i + 1]
        ax.imshow(reconstruction.reshape(64, 64), cmap='gray')
        ax.axis('off')
        ax.set_title(f'{n} components')

    for j in range(len(num_components_list) + 1, max(num_eigenfaces_individual, len(num_components_list) + 1)):
        axes[1, j].axis('off')

    plt.tight_layout()
    plt.show()
