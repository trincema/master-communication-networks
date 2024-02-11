import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def neyman_pearson_gender_height_classification(height_sample, male_mean, female_mean, alpha):
    # Calculate the sample mean and standard deviation
    sample_mean = np.mean(height_sample)
    sample_std = np.std(height_sample, ddof=1)  # Use ddof=1 for unbiased estimate
    print("Sample mean: " + str(sample_mean))
    print("Sample std: " + str(sample_std))

    # Compute the likelihood ratio test statistic
    likelihood_ratio = (norm.pdf(height_sample, female_mean, sample_std) /
                        norm.pdf(height_sample, male_mean, sample_std)).prod()

    # Determine the threshold value for the given significance level alpha
    threshold = norm.ppf(1 - alpha, loc=male_mean, scale=sample_std)
    print("threshold: " + str(threshold))

    # Perform the test
    gender_classified_as_male = likelihood_ratio <= threshold
    return gender_classified_as_male

# Example data for two gender populations
np.random.seed(42)
male_mean = 175  # mean height for males
female_mean = 162  # mean height for females
sample_size = 30

# Sample from the male population
sample_male = np.random.normal(male_mean, 5, sample_size)
print("sample_male: " + str(sample_male))
# Sample from the female population
sample_female = np.random.normal(female_mean, 5, sample_size)
print("sample_female: " + str(sample_female))

# Set the significance level
alpha = 0.05

# Perform Neyman-Pearson test
result_male = neyman_pearson_gender_height_classification(sample_male, male_mean, female_mean, alpha)
result_female = neyman_pearson_gender_height_classification(sample_female, male_mean, female_mean, alpha)

# Print the results
print("Classified as Male?", result_male)
print("Classified as Male?", not result_female)  # Not classified as male means classified as female

# PLOT THE SAMPLES
# Combine the samples for plotting
all_samples = [sample_male, sample_female]
# Create labels for the box plot
labels = ['Male', 'Female']

# Plot the box plot
plt.boxplot(all_samples, labels=labels)
plt.title('Height Samples for Male and Female Populations')
plt.xlabel('Gender')
plt.ylabel('Height (cm)')

# Plot all samples with different colors for males and females
for i, samples in enumerate(all_samples):
    color = 'blue' if i == 0 else 'pink'  # Different colors for males and females
    plt.scatter([i + 1] * len(samples), samples, color=color, label=labels[i])

plt.legend()
plt.show()