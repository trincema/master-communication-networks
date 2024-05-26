import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images
train_images_flat = train_images.reshape(-1, 28*28)
test_images_flat = test_images.reshape(-1, 28*28)

# Determine the optimal K from previous cross-validation results
optimal_k = 3  # Replace this with the actual best K found from cross-validation

# Train the KNN model with the optimal K
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(train_images_flat, train_labels)

# Predict the labels for the test set
test_predictions = knn.predict(test_images_flat)

# Compute the confusion matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title(f'Confusion Matrix for KNN with K={optimal_k}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate and print the overall accuracy
accuracy = accuracy_score(test_labels, test_predictions)
print(f'Overall classification accuracy: {accuracy:.4f}')
