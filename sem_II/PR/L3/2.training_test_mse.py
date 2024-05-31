import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Set label size for plots
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

# Load data
data = np.genfromtxt('diabetes-data.csv', delimiter=",")
features = ['age', 'sex', 'bmi', 'blood pressure', 'serum1', 'serum2', 'serum3', 'serum4', 'serum5', 'serum6']
x = data[:, 0:10]  # predictors
y = data[:, 10]    # response variable

def split_data(n_train):
    if (n_train < 0) or (n_train > 442):
        print("Invalid number of training points")
        return
    np.random.seed(0)
    perm = np.random.permutation(442)
    training_indices = perm[range(0, n_train)]
    test_indices = perm[range(n_train, 442)]
    trainx = x[training_indices, :]
    trainy = y[training_indices]
    testx = x[test_indices, :]
    testy = y[test_indices]
    return trainx, trainy, testx, testy

# Function to train and evaluate model
def train_and_evaluate(n_train):
    trainx, trainy, testx, testy = split_data(n_train)
    regr = linear_model.LinearRegression()
    regr.fit(trainx, trainy)
    
    # Make predictions
    train_pred = regr.predict(trainx)
    test_pred = regr.predict(testx)
    
    # Calculate MSE
    train_mse = mean_squared_error(trainy, train_pred)
    test_mse = mean_squared_error(testy, test_pred)
    
    return train_mse, test_mse

# Training set sizes to evaluate
training_sizes = [20, 50, 100, 200]
results = []

# Calculate MSE for each training size
for n_train in training_sizes:
    train_mse, test_mse = train_and_evaluate(n_train)
    results.append((n_train, train_mse, test_mse))
    print(f"Training size: {n_train}")
    print(f"Training MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

# Plot results
training_sizes, train_mses, test_mses = zip(*results)
plt.plot(training_sizes, train_mses, 'o-', label='Training MSE')
plt.plot(training_sizes, test_mses, 'o-', label='Test MSE')
plt.xlabel('Training Set Size', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.legend()
plt.show()
