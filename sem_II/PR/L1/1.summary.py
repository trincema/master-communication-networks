import pandas as pd
# Load dataset from CSV file
data = pd.read_csv('Advertising.csv')
# Compute summary statistics
summary = data.describe()
print(summary)