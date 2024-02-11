% CRLB for estimating the mean of a normal distribution

% Parameters
trueMean = 5;           % True mean of the normal distribution
sampleSize = 100;       % Size of the sample
sigma = 2;              % Known standard deviation

% Generate a random sample from the normal distribution
data = trueMean + sigma * randn(sampleSize, 1);

% Likelihood function for the normal distribution
likelihood = @(theta) normpdf(data, theta, sigma);

% Derivative of the log-likelihood function
dLogLikelihood = @(theta) (data - theta) / sigma^2;

% Fisher Information for estimating the mean of a normal distribution
fisherInformation = -mean(dLogLikelihood(trueMean)^2);

% Cramér-Rao Lower Bound for the variance of an unbiased estimator
crlbVariance = 1 / fisherInformation;

% Display results
fprintf('True mean: %.2f\n', trueMean);
fprintf('Sample mean: %.2f\n', mean(data));
fprintf('CRLB Variance: %.4f\n', crlbVariance);
