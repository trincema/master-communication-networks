import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset from CSV file
data = pd.read_csv('Advertising.csv')

# Define Features and Targets
feature_cols = ['TV', 'Radio', 'Newspaper']
target_col = 'Sales'

# Create a seaborn pairplot
sns.pairplot(data, x_vars=feature_cols, y_vars=target_col, height=7, aspect=0.7, kind='reg')
plt.show()