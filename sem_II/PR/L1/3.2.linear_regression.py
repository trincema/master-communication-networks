# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Advertising.csv')

# Define the feature columns and the target column
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]  # Features
y = data['Sales']       # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
linreg = LinearRegression()
# Train the model using the training data
linreg.fit(X_train, y_train)
# Make predictions using the testing data
y_pred = linreg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# Optionally, you can print the coefficients
print(f"Intercept: {linreg.intercept_}")
print(f"Coefficients: {linreg.coef_}")

# Visualize the results using matplotlib (optional)
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()
