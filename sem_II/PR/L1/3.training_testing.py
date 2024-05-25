# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Advertising.csv')
# Define the feature columns and the target column
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]  # Features
y = data['Sales']       # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Training Labels Shape: {y_train.shape}")
print(f"Testing Labels Shape: {y_test.shape}")

# Display the training and testing datasets
print("Features Train Dataset")
print(X_train)
print("Features Test Dataset")
print(X_test)
print("Target Train Dataset")
print(y_train)
print("Target Test Dataset")
print(y_test)

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