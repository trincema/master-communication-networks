import sys
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from time import time

# Load the dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images
train_images = train_images.reshape(-1, 28*28)
test_images = test_images.reshape(-1, 28*28)

# Create the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Measure the time taken to train the model
start_time = time()
knn.fit(train_images, train_labels)
training_time = time() - start_time
print(f"Training time: {training_time:.2f} seconds")

# Predict on the test set
start_time = time()
test_predictions = knn.predict(test_images)
prediction_time = time() - start_time
print(f"Prediction time: {prediction_time:.2f} seconds")

# Evaluate the model
accuracy = accuracy_score(test_labels, test_predictions)
print(f"KNN Classification Accuracy: {accuracy:.4f}")

# Estimate memory usage for the model parameters
memory_usage_model = sys.getsizeof(knn)

# Estimate memory usage for the training data
memory_usage_training_data = train_images.nbytes + train_labels.nbytes
# Print memory consumption
print(f"Memory consumption for training data: {memory_usage_training_data / (1024 * 1024):.2f} MB")
print(f"Memory consumption for KNN model: {memory_usage_model / (1024 * 1024):.2f} MB")