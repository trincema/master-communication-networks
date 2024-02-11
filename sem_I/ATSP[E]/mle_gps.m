% True position (latitude, longitude)
true_position = [37.7749, -122.4194];

% Simulate observed distances from satellites
num_satellites = 4;
true_distances = normrnd(true_position, 0.01);

% Add some noise to the distances
measurement_noise = 0.01;
noisy_distances = true_distances + normrnd(0, measurement_noise, 1, num_satellites);

% Define the likelihood function for GPS positioning
likelihood = @(position) -0.5 * sum(((noisy_distances - vecnorm(repmat(position, num_satellites, 1) - true_position, 2, 2)) / measurement_noise).^2);

% Initial guess for position
initial_position = [37.775, -122.42];

% Minimize the negative log-likelihood to obtain MLE estimates
options = optimset('fminsearch');
mle_position = fminsearch(likelihood, initial_position, options);

% Plotting
figure;

% Plot true position
scatter(true_position(2), true_position(1), 100, 'g', '*', 'DisplayName', 'True Position');
hold on;

% Plot observed distances
scatter(true_position(2) + randn(1, num_satellites) * 0.005, true_position(1) + randn(1, num_satellites) * 0.005, ...
    'b', 'o', 'filled', 'DisplayName', 'Observed Satellites');

% Plot MLE estimated position
scatter(mle_position(2), mle_position(1), 100, 'r', 'x', 'DisplayName', 'MLE Estimated Position');

% Draw lines connecting true position and observed satellites
for i = 1:num_satellites
    plot([true_position(2), true_position(2) + randn * 0.005], ...
         [true_position(1), true_position(1) + randn * 0.005], 'Color', [0.7 0.7 0.7], 'LineStyle', '--');
end

% Draw lines connecting MLE estimated position and observed satellites
for i = 1:num_satellites
    plot([mle_position(2), true_position(2) + randn * 0.005], ...
         [mle_position(1), true_position(1) + randn * 0.005], 'Color', [1 0.5 0], 'LineStyle', '--');
end

% Add labels and legend
xlabel('Longitude');
ylabel('Latitude');
title('GPS Positioning Example with MLE');
legend;

% Hold off to end plotting
hold off;
