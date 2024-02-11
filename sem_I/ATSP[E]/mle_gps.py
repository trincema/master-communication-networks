import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# True position (latitude, longitude)
true_position = np.array([37.7749, -122.4194])

# Simulate observed distances from satellites
num_satellites = 4
true_distances = np.linalg.norm(np.random.randn(num_satellites, 2) * 0.01 + true_position, axis=1)

# Add some noise to the distances
measurement_noise = 0.01
noisy_distances = true_distances + np.random.normal(0, measurement_noise, num_satellites)

# Define the likelihood function for GPS positioning
def likelihood(position, distances):
    estimated_distances = np.linalg.norm(np.tile(position, (num_satellites, 1)) - true_position, axis=1)
    log_likelihood = -0.5 * np.sum(((distances - estimated_distances) / measurement_noise) ** 2)
    return -log_likelihood  # We minimize the negative log-likelihood

# Initial guess for position
initial_position = np.array([37.775, -122.42])

# Minimize the negative log-likelihood to obtain MLE estimates
result = minimize(likelihood, initial_position, args=(noisy_distances,), method='Nelder-Mead')

# Extract MLE estimates for position
mle_position = result.x

# Plotting
plt.figure(figsize=(12, 6))

# Plot true position
plt.scatter(true_position[1], true_position[0], color='green', marker='*', label='True Position')

# Plot observed distances
plt.scatter(true_position[1] + np.random.normal(0, 0.005, num_satellites),
            true_position[0] + np.random.normal(0, 0.005, num_satellites),
            color='blue', marker='o', label='Observed Satellites')

# Plot MLE estimated position
plt.scatter(mle_position[1], mle_position[0], color='red', marker='x', label='MLE Estimated Position')

# Draw lines connecting true position and observed satellites
for i in range(num_satellites):
    plt.plot([true_position[1], true_position[1] + np.random.normal(0, 0.005)],
             [true_position[0], true_position[0] + np.random.normal(0, 0.005)], color='gray', linestyle='--')

# Draw lines connecting MLE estimated position and observed satellites
for i in range(num_satellites):
    plt.plot([mle_position[1], true_position[1] + np.random.normal(0, 0.005)],
             [mle_position[0], true_position[0] + np.random.normal(0, 0.005)], color='orange', linestyle='--')

# Add labels and legend
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('GPS Positioning Example with MLE')
plt.legend()

# Show the plot
plt.show()
