import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# Calculate the least squares estimators
X_mean = np.mean(X)
Y_mean = np.mean(Y)

beta_1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)
beta_0 = Y_mean - beta_1 * X_mean

# Make predictions using the estimated parameters
Y_pred = beta_0 + beta_1 * X

# Plot the data and the linear regression line
plt.scatter(X, Y, label='Original data')
plt.plot(X, Y_pred, color='red', label='Linear regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression using Least Squares Estimation')
plt.legend()
plt.show()

# Print the estimated parameters
print(f'Estimated intercept (beta_0): {beta_0}')
print(f'Estimated slope (beta_1): {beta_1}')
