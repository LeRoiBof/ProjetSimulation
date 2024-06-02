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
    index = 0
    while True:
        if index + slice_size > len(pi_decimals):
            index = 0
        slice = pi_decimals[index:index + slice_size]
        index += slice_size
        random_number = int(slice)
        yield random_number / 10 ** slice_size


def gap_test(sequence, alpha, beta, t, n):
    print(f"Sequence: {sequence}")
    # Initialize
    count = [0] * (t + 1)
    j = -1
    s = 0

    # Calculate p
    p = beta - alpha

    while s < n:
        print(f"Step {s}")
        # Set T zero
        r = 0
        while True:
            # Increase j
            j += 1
            if j >= len(sequence):
                print(f"Reached end of sequence at step {s}.")
                break

            # Check if sequence[j] is in the range [alpha, beta)
            print(type(sequence[j]))
            if alpha <= sequence[j] < beta:
                break
            else:
                print(f"Sequence[{j}] = {sequence[j]} is not in the range [{alpha}, {beta}).")
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


def maximum_test():
    #TODO: Implement the maximum test
    pass

def chisquare_test():
    #TODO: Implement the chi-square test
    pass

if __name__ == "__main__":
    # Get the decimals of pi
    pi_decimals = get_pi_decimals("pi_decimals_single_line.txt")

    # Generate random numbers based on the decimals of pi
    random_number_generator = generate_random_numbers(pi_decimals, 1)

    result = gap_test([next(random_number_generator) for _ in range(1000)], 0.2, 0.5, 5, 1000)
    print(f"Observed counts: {result['observed_counts']}")
    print(f"Expected counts: {result['expected_counts']}")
    print(f"Chi-square statistic: {result['chi_square_statistic']}")
    print(f"Critical value: {result['critical_value']}")
    print(f"Reject null hypothesis: {result['reject_null']}")
