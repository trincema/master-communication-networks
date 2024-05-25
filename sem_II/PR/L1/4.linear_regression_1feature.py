import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from CSV file
data = pd.read_csv('Advertising.csv')
feature_cols = ['TV', 'Radio', 'Newspaper']

# Function to perform linear regression, calculate RMSE, and plot the results
def evaluate_and_plot_model(feature):
    X = data[[feature]]
    y = data['Sales']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Train the linear regression model
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    
    # Predict the target variable on the test set
    y_pred = lm.predict(X_test)
    
    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
    plt.scatter(X_test, y_pred, color='red', label='Predicted Sales')
    plt.plot(X_test, y_pred, color='black', linewidth=2)
    plt.xlabel(feature)
    plt.ylabel('Sales')
    plt.title(f'Regression Model using {feature} (RMSE: {rmse:.2f})')
    plt.legend()
    plt.show()
    
    return rmse

# Evaluate each model, print the RMSE, and plot the results
for feature in feature_cols:
    rmse = evaluate_and_plot_model(feature)
    print(f'RMSE for {feature}: {rmse:.2f}')
