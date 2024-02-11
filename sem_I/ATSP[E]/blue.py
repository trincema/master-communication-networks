import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Generate synthetic data for a simple linear regression model
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Independent variable
epsilon = np.random.randn(100, 1) * 2  # Normally distributed errors
beta_0 = 5  # True intercept
beta_1 = 2  # True slope
y_true = beta_0 + beta_1 * X.squeeze() + epsilon.squeeze()  # True dependent variable

# Add a constant term for the intercept
X_with_intercept = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y_true, X_with_intercept)
results = model.fit()

# Extract coefficients (intercept and slope)
beta_hat = results.params

# Plot the true data and the regression line
plt.scatter(X, y_true, label='True Data')
plt.plot(X, beta_hat[0] + beta_hat[1] * X, color='red', label='Regression Line (BLUE)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()

# Display the estimated coefficients
print(f'Intercept (beta_0): {beta_hat[0]}')
print(f'Slope (beta_1): {beta_hat[1]}')
