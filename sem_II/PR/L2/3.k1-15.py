import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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

# List to store accuracies for different values of K
accuracies = []

# Iterate over values of K from 1 to 15
for k in range(1, 16):
    print("k-" + str(k))
    # Create the KNN model with the current value of K
    knn = KNeighborsClassifier(n_neighbors=k)
    # Train the model
    knn.fit(train_images_flat, train_labels)
    # Predict on the test set
    test_predictions = knn.predict(test_images_flat)
    # Calculate the accuracy
    accuracy = accuracy_score(test_labels, test_predictions)
    # Append the accuracy to the list
    accuracies.append(accuracy)

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(1, 16), accuracies, marker='o')
plt.title('KNN Classification Accuracy for Different Values of K')
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

print(accuracies)

# Find the best value for K
best_k = np.argmax(accuracies) + 1
best_accuracy = accuracies[best_k - 1]

print(f'The best value for K is {best_k} with an accuracy of {best_accuracy:.4f}')
