import numpy as np
import statsmodels.api as sm

# Simulate some data
np.random.seed(42)
X = np.random.rand(100, 1)  # Independent variable
epsilon = np.random.normal(0, 1, size=(100, 1))  # Error term
beta_0, beta_1 = 2, 3  # True parameters
y = beta_0 + beta_1 * X + epsilon  # Dependent variable
print(X)
print(epsilon)
print(y)

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit()

# Display regression results
print(results.summary())