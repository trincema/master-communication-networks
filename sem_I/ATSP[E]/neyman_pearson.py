import numpy as np
from scipy.stats import norm
 
# Generate sample data
np.random.seed(0)
n = 100  # Number of samples
h0 = 0  # Null hypothesis mean
h1 = 1  # Alternative hypothesis mean
sigma = 1  # Common standard deviation
data = np.random.randn(n) + h1  # Samples from the alternative distribution
print('data: ' + str(data))
 
# Compute likelihood ratio test
likelihood_ratio = np.prod(norm.pdf(data, h1, sigma)) / np.prod(norm.pdf(data, h0, sigma))
print("Likelihood ratio = ", likelihood_ratio)
 
 
# Set significance level and threshold
alpha = 0.05  # Significance level
threshold = norm.ppf(1 - alpha)  # Threshold based on significance level
print("Threshold, k = ", threshold)
 
# Perform hypothesis test
if likelihood_ratio > threshold:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis")