import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Advertising.csv')

# Define the feature pairs
feature_pairs = [
    ['TV', 'Radio'],
    ['TV', 'Newspaper'],
    ['Radio', 'Newspaper']
]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['TV', 'Radio', 'Newspaper']], data['Sales'], test_size=0.2, random_state=1)

# Function to perform linear regression and calculate RMSE
def evaluate_model(features):
    model = LinearRegression()
    model.fit(X_train[features], y_train)
    predictions = model.predict(X_test[features])
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse, model

# Evaluate each feature pair
results = {}
for pair in feature_pairs:
    rmse, model = evaluate_model(pair)
    results[tuple(pair)] = rmse

# Print the RMSE values for each model
for pair, rmse in results.items():
    print(f"RMSE for {pair}: {rmse:.2f}")

# Plotting the results
pairs = list(results.keys())
rmses = list(results.values())

plt.figure(figsize=(10, 6))
plt.barh([str(pair) for pair in pairs], rmses, color='skyblue')
plt.xlabel('RMSE')
plt.ylabel('Feature Pairs')
plt.title('RMSE for Different Feature Pairs')
plt.show()
