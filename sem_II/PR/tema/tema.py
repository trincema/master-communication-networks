import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import skimage
print(skimage.__version__)
from skimage.feature import graycomatrix, graycoprops

# Load the image
image = io.imread('testimg.bmp')

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Convert the grayscale image to uint8 type
gray_image = (gray_image * 255).astype('uint8')

# Define the distances and angles
distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Compute the GLCM
glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

# Extract texture features
contrast = graycoprops(glcm, prop='contrast')
dissimilarity = graycoprops(glcm, prop='dissimilarity')
homogeneity = graycoprops(glcm, prop='homogeneity')
energy = graycoprops(glcm, prop='energy')
correlation = graycoprops(glcm, prop='correlation')

# Display the GLCM and texture features
print("GLCM Shape: ", glcm.shape)
print("Contrast:\n", contrast)
print("Dissimilarity:\n", dissimilarity)
print("Homogeneity:\n", homogeneity)
print("Energy:\n", energy)
print("Correlation:\n", correlation)

# Visualize the GLCM for the first distance and angle
plt.figure(figsize=(10, 8))
for i, distance in enumerate(distances):
    for j, angle in enumerate(angles):
        plt.subplot(len(distances), len(angles), i * len(angles) + j + 1)
        plt.imshow(glcm[:, :, i, j], cmap='gray', interpolation='nearest')
        plt.title(f'd={distance}, θ={angle*180/np.pi:.1f}°')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Pixel Intensity')

plt.tight_layout()
plt.show()
