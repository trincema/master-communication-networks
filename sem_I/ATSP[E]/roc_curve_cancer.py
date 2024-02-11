import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Simulate data for patients with and without cancer
np.random.seed(42)

# Biomarker levels for patients without cancer (negative class)
biomarker_negative = np.random.normal(10, 2, 100)

# Biomarker levels for patients with cancer (positive class)
biomarker_positive = np.random.normal(15, 2, 100)

# Combine samples and create corresponding labels (0 for negative, 1 for positive)
biomarker_data = np.concatenate([biomarker_negative, biomarker_positive])
labels = np.concatenate([np.zeros_like(biomarker_negative), np.ones_like(biomarker_positive)])

# Shuffle the data
indices = np.arange(len(biomarker_data))
np.random.shuffle(indices)
biomarker_data = biomarker_data[indices]
labels = labels[indices]

# Assume a diagnostic test that assigns a positive result if biomarker level exceeds a threshold
threshold = 12
test_results = (biomarker_data > threshold).astype(int)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(labels, test_results)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Cancer Marker Test')
plt.legend(loc="lower right")
plt.show()
