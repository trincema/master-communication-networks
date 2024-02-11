# Suppose you have a dataset and you want to find the mean (μ) and standard deviation (σ)
# that maximize the likelihood of observing that specific dataset under the assumption of a normal distribution.
# We generate a sample dataset from a normal distribution with a known mean (loc=5) and standard deviation (scale=2).
# The likelihood function calculates the log-likelihood of the data given the parameters (mu and sigma) assuming a normal distribution.
# The minimize function from scipy.optimize is used to minimize the negative log-likelihood, effectively finding the maximum likelihood estimates.
# The MLE estimates for the mean and standard deviation are then printed.
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Generate a sample dataset from a normal distribution
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=100)
print("data: " + str(data))

# Define the likelihood function for a normal distribution
def likelihood(parameters, data):
    mu, sigma = parameters
    log_likelihood = np.sum(norm.logpdf(data, loc=mu, scale=sigma))
    return -log_likelihood  # We minimize the negative log-likelihood

# Initial guess for parameters
initial_parameters = [0, 1]

# Minimize the negative log-likelihood to obtain MLE estimates
result = minimize(likelihood, initial_parameters, args=(data,), method='Nelder-Mead')
print("result: " + str(result))

# Extract MLE estimates
mle_mu, mle_sigma = result.x
print("mle_mu: " + str(mle_mu) + " mle_sigma: " + str(mle_sigma))

# Plot the histogram of the dataset
plt.hist(data, bins=20, density=True, alpha=0.5, color='blue', label='Histogram of Data')

# Plot the estimated normal distribution based on MLE
x_range = np.linspace(min(data), max(data), 100)
pdf_values = norm.pdf(x_range, loc=mle_mu, scale=mle_sigma)
plt.plot(x_range, pdf_values, 'r-', label='Estimated Normal Distribution (MLE)')

# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Histogram and Estimated Normal Distribution')
plt.legend()

# Show the plot
plt.show()
