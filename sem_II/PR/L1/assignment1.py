import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset from CSV file
data = pd.read_csv('Advertising.csv')
# Compute summary statistics
summary = data.describe()
print(summary)

feature_cols = ['TV', 'Radio', 'Newspaper']
target_col = 'Sales'

# Create a seaborn pairplot
#sns.load_dataset(data)
sns.pairplot(data, x_vars=feature_cols, y_vars=target_col, height=7, aspect=0.7, kind='reg')
plt.show()

# Splitting X and y into training and testing sets
# default split is 75% for trining and 25% for testing
X = data[feature_cols]
y = data.Sales
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X)
print(y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# perform the linear regression using “scikit-learn” library
linearRegression = LinearRegression()
# fit the model to the training data (learn the coefficients)
linearRegression.fit(X_train, y_train)
# print the coefficients
print("Intercept: " + str(linearRegression.intercept_))
print("Coefficients: " + str(linearRegression.coef_))

# Calculate RMSE mathematically (using numpy)
# rmse = np.sqrt()