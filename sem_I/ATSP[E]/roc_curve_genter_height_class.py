import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Example data for two gender populations
np.random.seed(42)
male_mean = 175  # mean height for males
female_mean = 162  # mean height for females
sample_size = 30

# Sample from the male population
sample_male = np.random.normal(male_mean, 5, sample_size)
print("sample_male: " + str(sample_male))
# Sample from the female population
sample_female = np.random.normal(female_mean, 5, sample_size)
print("sample_female: " + str(sample_female))

# Combine samples and create corresponding labels (1 for male, 0 for female)
all_samples = np.concatenate([sample_male, sample_female])
labels = np.concatenate([np.ones_like(sample_male), np.zeros_like(sample_female)])

# Randomly shuffle the data to simulate a binary classification problem
indices = np.arange(len(all_samples))
np.random.shuffle(indices)
all_samples = all_samples[indices]
labels = labels[indices]

# Train a simple classifier (for illustration purposes)
# In a real-world scenario, you would use a proper machine learning model
threshold = (male_mean + female_mean) / 2
predictions = (all_samples > threshold).astype(int)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(labels, predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
