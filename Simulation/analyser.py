import random

import numpy as np
from scipy.stats import chisquare, kstest

# This function reads a file containing the decimals of pi and returns them as a list of integers.
def get_pi_decimals(file):
    with open(file, 'r') as f:
        pi_decimals = f.read().strip()
    return pi_decimals


# This function generates random numbers based on the decimals of pi.
# It yields a new random number each time it is called.
def generate_random_numbers(pi_decimals, slice_size):
    index = 0
    while True:
        if index + slice_size > len(pi_decimals):
            index = 0
        slice = pi_decimals[index:index + slice_size]
        index += slice_size
        random_number = int(slice)
        yield random_number / 10 ** slice_size

# This function tests a random number generator using the Chi-square and Kolmogorov-Smirnov tests.
# It returns the test statistics and p-values for both tests.
def test_generator(generator, n_samples):
    samples = [next(generator) for _ in range(n_samples)]
    observed_frequencies, _ = np.histogram(samples, bins=10)
    expected_frequencies = np.ones(10) * n_samples / 10
    chi2_stat, chi2_p = chisquare(observed_frequencies, expected_frequencies)
    ks_stat, ks_p = kstest(samples, 'uniform')
    return chi2_stat, chi2_p, ks_stat, ks_p

# The main part of the script tests two random number generators: one based on the decimals of pi, and Python's default generator.
# It prints the results of the Chi-square and Kolmogorov-Smirnov tests for both generators.
if __name__ == "__main__":
    n_samples = 1000000
    pi_decimals = get_pi_decimals('pi_decimals_single_line.txt')

    for slice_size in range(1, 11):
        custom_generator = generate_random_numbers(pi_decimals, slice_size)
        chi2_stat, chi2_p, ks_stat, ks_p = test_generator(custom_generator, n_samples)
        print(f"Custom generator (slice size {slice_size}):")
        print(f"Chi2 test : statistic={chi2_stat}, p-value={chi2_p}")
        print(f"KS test : statistic={ks_stat}, p-value={ks_p}")

    for slice_size in range(1, 11):
        python_generator = iter(lambda: random.uniform(0, 1), None)
        chi2_stat, chi2_p, ks_stat, ks_p = test_generator(python_generator, n_samples)
        print(f"Custom generator (slice size {slice_size}):")
        print(f"Chi2 test : statistic={chi2_stat}, p-value={chi2_p}")
        print(f"KS test : statistic={ks_stat}, p-value={ks_p}")