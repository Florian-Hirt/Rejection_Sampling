import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon


# Function to generate samples from different original distributions
def original_distribution(num_samples, distribution_type):
    if distribution_type == 'uniform':
        return np.random.choice(range(10), num_samples)
    elif distribution_type == 'gaussian':
        return np.round(norm.rvs(loc=5, scale=2, size=num_samples))
    elif distribution_type == 'exponential':
        return np.round(expon.rvs(scale=2, size=num_samples))
    else:
        raise ValueError("Invalid distribution type")


# Weight function
def weight_function(i):
    return i + 1  # Some arbitrary weight function


# Rejection sampling to construct and sample from the weighted distribution D(w)
def rejection_sampling(num_samples, distribution_type, weight_function):
    samples = []
    weights = []
    W = np.max(weight_function(np.arange(10)))  # Supremum of the weight function
    while len(samples) < num_samples:
        candidate = original_distribution(1, distribution_type)[0]
        acceptance_probability = weight_function(candidate) / W
        if np.random.rand() < acceptance_probability:
            samples.append(candidate)
            weights.append(weight_function(candidate))
    return np.array(samples), np.array(weights)


# Generate and plot samples from different original distributions
distribution_types = ['uniform', 'gaussian', 'exponential']
num_samples = 10000

plt.figure(figsize=(18, 8))

plt.subplot(3, 3, 2)
plt.plot(range(10), weight_function(np.arange(10)), marker='o', linestyle='-', color='r')
plt.title('Weight Function (w(i))')

# Plot original distributions
for i, distribution_type in enumerate(distribution_types):
    plt.subplot(3, 3, i + 4)
    original_samples = original_distribution(num_samples, distribution_type)
    plt.hist(original_samples, bins=range(11), edgecolor='black', alpha=0.7)
    plt.title(f'{distribution_type.capitalize()} Distribution (D)')

# Plot re-weighted distributions
for i, distribution_type in enumerate(distribution_types):
    plt.subplot(3, 3, i + 7)
    weighted_samples, weights = rejection_sampling(num_samples, distribution_type, weight_function)
    plt.hist(weighted_samples, bins=range(11), edgecolor='black', alpha=0.7)
    plt.title(f'{distribution_type.capitalize()} Distribution (D(w))')

plt.tight_layout()
plt.show()
