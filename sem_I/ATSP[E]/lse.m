rng(42); % Set seed for reproducibility

% Generate synthetic data
X = 2 * rand(100, 1);
Y = 4 + 3 * X + randn(100, 1);
print("X: " + str(X))
print("Y: " + str(Y))

% Calculate the least squares estimators
X_mean = mean(X);
Y_mean = mean(Y);
print("X_mean: " + str(X_mean))
print("Y_mean: " + str(Y_mean))

beta_1 = sum((X - X_mean) .* (Y - Y_mean)) / sum((X - X_mean).^2);
beta_0 = Y_mean - beta_1 * X_mean;
print("beta_1: " + str(beta_1))
print("beta_0: " + str(beta_0))

% Make predictions using the estimated parameters
Y_pred = beta_0 + beta_1 * X;
print("Y_pred: " + str(Y_pred))

% Plot the data and the linear regression line
scatter(X, Y, 'DisplayName', 'Original data');
hold on;
plot(X, Y_pred, 'r', 'DisplayName', 'Linear regression');
xlabel('X');
ylabel('Y');
title('Linear Regression using Least Squares Estimation');
legend('show');
hold off;

% Print the estimated parameters
fprintf('Estimated intercept (beta_0): %.4f\n', beta_0);
fprintf('Estimated slope (beta_1): %.4f\n', beta_1);