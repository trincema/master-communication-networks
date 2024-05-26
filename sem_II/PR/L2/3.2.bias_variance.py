import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images
train_images_flat = train_images.reshape(-1, 28*28)
test_images_flat = test_images.reshape(-1, 28*28)

# List to store results for different values of K
biases = []
variances = []
ks = list(range(1, 16))  # Convert to list for easier indexing

# Number of subsets for estimating bias and variance
num_subsets = 10
subset_size = len(train_images_flat) // num_subsets

for k in ks:
    print("Processing " + str(k) + "...")
    subset_predictions = []

    for i in range(num_subsets):
        # Generate a random subset of training data
        subset_indices = np.random.choice(len(train_images_flat), subset_size, replace=False)
        subset_train_images = train_images_flat[subset_indices]
        subset_train_labels = train_labels[subset_indices]

        # Create the KNN model with the current value of K
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Train the model
        knn.fit(subset_train_images, subset_train_labels)
        
        # Predict on the test set
        test_predictions = knn.predict(test_images_flat)
        subset_predictions.append(test_predictions)
    
    # Convert to numpy array for easy calculations
    subset_predictions = np.array(subset_predictions)
    
    # Calculate the average prediction for each test point
    avg_predictions = np.mean(subset_predictions, axis=0)
    
    # Calculate bias
    bias = np.mean((avg_predictions - test_labels) ** 2)
    biases.append(bias)
    
    # Calculate variance
    variance = np.mean(np.var(subset_predictions, axis=0))
    variances.append(variance)

print("Biases: " + str(biases))
print("Variances: " + str(variances))

# Plot bias and variance
plt.figure(figsize=(10, 6))
plt.plot(ks, biases, marker='o', label='Bias')
plt.plot(ks, variances, marker='o', label='Variance')
plt.title('Bias and Variance for Different Values of K')
plt.xlabel('Value of K')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

# Find the best value for K
total_errors = np.array(biases) + np.array(variances)
best_k_index = np.argmin(total_errors)
print("Total errors: " + str(total_errors))

# Print the best value for K
if best_k_index < len(ks):
    best_k = ks[best_k_index]
    print(f'The best value for K is {best_k}')
else:
    print("Error: Best K index is out of range")
