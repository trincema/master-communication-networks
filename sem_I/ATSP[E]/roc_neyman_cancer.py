# Certainly! The Neyman-Pearson framework involves setting a specific operating point (threshold) based on
# the trade-off between sensitivity and specificity. In the context of cancer markers, you might want to choose
# a threshold that maximizes sensitivity while keeping specificity at a desired level (or vice versa,
# depending on the application).
# We set a range of potential thresholds and evaluate the performance of the test at each threshold.
# Sensitivity and specificity are calculated for each threshold, and the points are plotted on the ROC curve.
# The curve illustrates the trade-off between sensitivity and specificity at different classification thresholds.
# You can analyze the curve and choose the operating point that aligns with your desired balance between sensitivity
# and specificity. The Neyman-Pearson framework helps in making decisions about the threshold based on the relative
# importance of false positives and false negatives in a given context.
import numpy as np
from sklearn.metrics import auc
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

# Set a range of potential thresholds
thresholds = np.linspace(min(biomarker_data), max(biomarker_data), 100)

# Initialize variables to store results
sensitivity_list = []
specificity_list = []

# Evaluate performance at each threshold
for threshold in thresholds:
    test_results = (biomarker_data > threshold).astype(int)

    # Calculate sensitivity and specificity
    true_positive = np.sum((test_results == 1) & (labels == 1))
    false_positive = np.sum((test_results == 1) & (labels == 0))
    true_negative = np.sum((test_results == 0) & (labels == 0))
    false_negative = np.sum((test_results == 0) & (labels == 1))

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)

    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)

# Compute AUC
roc_auc = auc(1 - np.array(specificity_list), sensitivity_list)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(1 - np.array(specificity_list), sensitivity_list, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Cancer Marker Test')
plt.legend(loc="lower right")
plt.show()
