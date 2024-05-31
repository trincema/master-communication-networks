# Standard includes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Routines for Linear Regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Set label size for plots
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

data = np.genfromtxt('diabetes-data.csv', delimiter=",")
features = ['age', 'sex', 'bmi', 'blood pressure', 'serum1', 'serum2', 'serum3', 'serum4', 'serum5', 'serum6']
x = data[:,0:10]    # predictors
y = data[:,10]      # response variable

def one_feature_regression(x, y, f):
    if (f < 0) or (f > 9):
        print("Feature index is out of bounds.")
        return
    regr = linear_model.LinearRegression()
    x1 = x[:,[f]]
    regr.fit(x1, y)
    # Make predictions using the model
    y_pred = regr.predict(x1)
    # Plot data points as well as predictions
    plt.plot(x1, y, 'bo')
    plt.plot(x1, y_pred, 'r-', linewidth=3)
    plt.xlabel(features[f], fontsize=14)
    plt.ylabel('Disease progression', fontsize=14)
    plt.show()
    print('Mean Square Error ' + features[f] + ': ', mean_squared_error(y, y_pred))
    return regr

def split_data(n_train):
    if (n_train < 0) or (n_train > 442):
        print("Invalid number of training points")
        return
    np.random.seed(0)
    perm = np.random.permutation(442)
    training_indices = perm[range(0, n_train)]
    test_indices = perm[range(n_train, 442)]
    trainx = x[training_indices,:]
    trainy = y[training_indices]
    testx = x[test_indices,:]
    testy = y[test_indices]
    return trainx, trainy, testx, testy

# Calculate MSE for each feature
mse_scores = []
for i in range(len(features)):
    mse = one_feature_regression(x, y, i)
    mse_scores.append((features[i], mse))
print('MSE scores: ' + str(mse_scores))

# Find the features with the lowest and second lowest MSE
sorted_mse = sorted(mse_scores, key=lambda item: item[1])
best_feature, best_mse = sorted_mse[0]
second_best_feature, second_best_mse = sorted_mse[1]

print(f"The feature with the lowest MSE is: {best_feature} with MSE = {best_mse}")
print(f"The second best feature is: {second_best_feature} with MSE = {second_best_mse}")