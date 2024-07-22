import json
import random
from collections import Counter

import numpy as np
from scipy.stats import chi2


# This function reads a file containing the decimals of pi and returns them as a list of integers.
def get_pi_decimals(file):
    with open(file, 'r') as f:
        pi_decimals = f.read().strip()
    return pi_decimals


# This function generates random numbers based on the decimals of pi.
# It yields a new random number each time it is called.
def generate_random_numbers(pi_decimals, slice_size):
    while True:
        index = random.randint(0, len(pi_decimals) - 1)
        if index + slice_size > len(pi_decimals):
            index = 0
        slice = pi_decimals[index:index + slice_size]
        random_number = int(slice)

        yield random_number / 10 ** slice_size


def gap_test(sequence, alpha, beta, t, n):
    """
    Perform a gap test on a sequence of numbers.

    The gap test is a statistical test that checks if the numbers in the sequence are uniformly distributed.

    Parameters:
    sequence (list): The sequence of numbers to test.
    alpha (float): The lower bound of the interval.
    beta (float): The upper bound of the interval.
    t (int): The maximum gap length to consider.
    n (int): The number of gaps to consider.

    Returns:
    dict: A dictionary containing the observed counts, expected counts, chi-square statistic, critical value, and whether to reject the null hypothesis.
    """
    # Initialize
    count = [0] * (t + 1)
    j = -1
    s = 0

    # Calculate p
    p = beta - alpha

    while s < n:
        # Set T zero
        r = 0
        while True:
            # Increase j
            j += 1
            if j >= len(sequence):
                break

            # Check if sequence[j] is in the range [alpha, beta)
            if alpha <= sequence[j] < beta:
                break
            else:
                # Increase r
                r += 1

        # Record gap length
        if r >= t:
            count[t] += 1
        else:
            count[r] += 1

        # Increase s
        s += 1

    # Calculate expected probabilities
    expected = [p * (1 - p) ** i for i in range(t)]
    expected.append((1 - p) ** t)

    # Calculate chi-square statistic
    observed = np.array(count)
    expected_counts = np.array(expected) * n
    chi_square_statistic = np.sum((observed - expected_counts) ** 2 / expected_counts)

    # Degrees of freedom
    degrees_of_freedom = t

    # Critical value
    critical_value = chi2.ppf(1 - 0.05, degrees_of_freedom)

    # Compare chi-square statistic with critical value
    reject_null = chi_square_statistic > critical_value

    return {
        "observed_counts": observed,
        "expected_counts": expected_counts,
        "chi_square_statistic": chi_square_statistic,
        "critical_value": critical_value,
        "reject_null": reject_null
    }


def calculate_runs(sequence):
    """
    Calculate the runs in a sequence.

    A run is defined as a sequence of increasing or decreasing numbers. The function calculates the length of each run.

    Parameters:
    sequence (list): The sequence of numbers to calculate runs for.

    Returns:
    list: A list of run lengths.
    """
    runs = []
    run_length = 1
    increasing = None

    for i in range(1, len(sequence)):
        if sequence[i] > sequence[i - 1]:
            if increasing is None or increasing:
                run_length += 1
                increasing = True
            else:
                runs.append(run_length)
                run_length = 1
                increasing = True
        elif sequence[i] < sequence[i - 1]:
            if increasing is None or not increasing:
                run_length += 1
                increasing = False
            else:
                runs.append(run_length)
                run_length = 1
                increasing = False
        else:
            if increasing is not None:
                runs.append(run_length)
                run_length = 1
                increasing = None

    runs.append(run_length)
    return runs


def compute_statistics(runs, n):
    """
    Compute the statistics for the runs test.

    The function calculates the V statistic, which is used in the runs test.

    Parameters:
    runs (list): The list of run lengths.
    n (int): The total number of runs.

    Returns:
    tuple: A tuple containing the V statistic, the observed counts, the expected means, and the covariance matrix.
    """
    t = 6
    count = [0] * (t + 1)

    for run in runs:
        if run >= t:
            count[t] += 1
        else:
            count[run] += 1

    # Coefficients (approximate values from provided matrix)
    a_matrix = np.array([
        [4529.4, 9044.9, 13568, 18091, 22615, 27892],
        [9044.9, 18097, 27139, 36187, 45234, 55789],
        [13568, 27139, 40721, 54281, 67852, 83685],
        [18091, 36187, 54281, 72414, 90470, 111580],
        [22615, 45234, 67852, 90470, 113262, 139476],
        [27892, 55789, 83685, 111580, 139476, 172860]
    ])

    b_vector = np.array([1/6, 5/24, 11/120, 19/720, 29/5040, 1/840])

    # Compute mean values
    means = np.array([n * b for b in b_vector])

    # Compute V statistic
    V = 0
    for i in range(t):
        for j in range(t):
            V += (count[i] - means[i]) * (count[j] - means[j]) * a_matrix[i, j]

    V /= (n - 6)

    return V, count, means, a_matrix

def poker_test(sequence, m):

    n = len(sequence)
    k = 10 ** m
    m = 5
    count = [0] * k

    for i in range(0, n, m):
        number = 0
        for j in range(m):
            number += sequence[i + j] * 10 ** (m - j - 1)
        count[number] += 1

    expected = n / k
    chi_square_statistic = sum((count[i] - expected) ** 2 / expected for i in range(k))

    degrees_of_freedom = k - 1
    critical_value = chi2.ppf(1 - 0.05, degrees_of_freedom)
    reject_null = chi_square_statistic > critical_value

    return {
        "observed_counts": count,
        "expected_counts": [expected] * k,
        "chi_square_statistic": chi_square_statistic,
        "critical_value": critical_value,
        "reject_null": reject_null
    }


def chisquare_test(sequence, expected_probs=None):
    """
    Perform a Chi-Square test on a sequence of numbers.

    The Chi-Square test is a statistical test that checks if the observed counts of numbers in the sequence
    are significantly different from the expected counts.

    Parameters:
    sequence (list): The sequence of numbers to test.

    Returns:
    dict: A dictionary containing the observed counts, expected counts, Chi-Square statistic, critical value,
    and whether to reject the null hypothesis.
    """
    # Count the occurrence of each number in the sequence
    observed_counts = Counter(sequence)
    n = len(sequence)
    k = len(observed_counts)

    # If expected_probs is not provided, assume a uniform distribution
    if expected_probs is None:
        unique_values = observed_counts.keys()
        expected_probs = {key: 1 / len(unique_values) for key in unique_values}

    # Calculate the expected counts for each number
    expected_counts = {key: n * expected_probs[key] for key in observed_counts.keys()}

    # Calculate the Chi-Square statistic
    chi_square_statistic = sum((observed_counts[key] - expected_counts[key]) ** 2 / expected_counts[key] for key in observed_counts.keys())

    # Calculate the degrees of freedom
    degrees_of_freedom = k - 1

    # Calculate the critical value
    critical_value = chi2.ppf(1 - 0.05, degrees_of_freedom)

    # Determine whether to reject the null hypothesis
    reject_null = chi_square_statistic > critical_value

    return {
        "observed_counts": observed_counts,
        "expected_counts": expected_counts,
        "chi_square_statistic": chi_square_statistic,
        "critical_value": critical_value,
        "reject_null": reject_null
    }

def custom_generator_test(pi_decimals):
    # Generate random numbers based on the decimals of pi
    random_number_generator = generate_random_numbers(pi_decimals, 6)

    # Perform a gap test on the generated random numbers
    # The gap test checks if the numbers in the sequence are uniformly distributed
    result = gap_test([next(random_number_generator) for _ in range(1000)], 0.2, 0.5, 5, 1000)

    # Write the result to a file
    with open("custom_generator_test_results.txt", "w") as f:
        f.write(f"----------- Gap test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")

    # Perform a runs test on the generated random numbers
    # The runs test checks if the numbers in the sequence are randomly distributed
    result = runs_test([next(random_number_generator) for _ in range(1000)])

    with open("custom_generator_test_results.txt", "a") as f:
        f.write(f"----------- Runs test -----------\n")
        f.write(f"V statistic: {result['V_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected means: {result['expected_means']}\n")

    # Perform a Chi-Square test on the generated random numbers
    # The Chi-Square test checks if the observed counts of numbers in the sequence are significantly different from the expected counts
    result = chisquare_test([next(random_number_generator) for _ in range(1000)])

    with open("custom_generator_test_results.txt", "a") as f:
        f.write(f"----------- Chi-Square test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")


def python_generator_test():
    # Generate random numbers using the Python random module
    random_numbers = [random.uniform(0, 1) for _ in range(1000)]

    # Perform a gap test on the generated random numbers
    # The gap test checks if the numbers in the sequence are uniformly distributed
    result = gap_test(random_numbers, 0.2, 0.5, 5, 1000)

    with open("python_generator_test_results.txt", "w") as f:
        f.write(f"----------- Gap test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")

    # Perform a runs test on the generated random numbers
    # The runs test checks if the numbers in the sequence are randomly distributed
    result = runs_test(random_numbers)

    with open("python_generator_test_results.txt", "a") as f:
        f.write(f"----------- Runs test -----------\n")
        f.write(f"V statistic: {result['V_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected means: {result['expected_means']}\n")

    # Perform a Chi-Square test on the generated random numbers
    # The Chi-Square test checks if the observed counts of numbers in the sequence are significantly different from the expected counts
    result = chisquare_test(random_numbers)

    with open("python_generator_test_results.txt", "a") as f:
        f.write(f"----------- Chi-Square test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")


def pi_decimals_test(pi_decimals):
    # Perform a gap test on the decimals of pi
    # The gap test checks if the numbers in the sequence are uniformly distributed
    result = gap_test([int(digit) for digit in pi_decimals], 0, 9, 5, 1000)

    with open("pi_decimals_test_results.txt", "w") as f:
        f.write(f"----------- Gap test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")

    # Perform a runs test on the decimals of pi
    # The runs test checks if the numbers in the sequence are randomly distributed
    result = runs_test([int(digit) for digit in pi_decimals])

    with open("pi_decimals_test_results.txt", "a") as f:
        f.write(f"----------- Runs test -----------\n")
        f.write(f"V statistic: {result['V_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected means: {result['expected_means']}\n")

    # Perform a Chi-Square test on the decimals of pi
    # The Chi-Square test checks if the observed counts of numbers in the sequence are significantly different from the expected counts
    result = chisquare_test([int(digit) for digit in pi_decimals])

    with open("pi_decimals_test_results.txt", "a") as f:
        f.write(f"----------- Chi-Square test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")



if __name__ == "__main__":
    # Get the decimals of pi from a file
    pi_decimals = get_pi_decimals("pi_decimals_single_line.txt")

    custom_generator_test(pi_decimals)

    python_generator_test()

    pi_decimals_test(pi_decimals)



