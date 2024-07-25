import json
import random
from collections import Counter
import math

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

# Fonction pour déterminer la catégorie de poker
def poker_category(rolls):
    counts = Counter(rolls).values()
    count_values = sorted(counts, reverse=True)
    if count_values[0] == 5:
        return "Poker"
    elif count_values[0] == 4:
        return "Carré"
    elif count_values[0] == 3 and count_values[1] == 2:
        return "Full"
    elif count_values[0] == 3:
        return "Brelan"
    elif count_values[0] == 2 and count_values[1] == 2:
        return "Double paire"
    elif count_values[0] == 2:
        return "Paire"
    else:
        return "Rien"

def stirling_number(n,k):
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    return k * stirling_number(n-1, k) + stirling_number(n-1, k-1)

def poker_probabilities(k, d):
    probabilities = []
    for r in range(1, k+1):
        S = stirling_number(k, r)
        assignments = math.factorial(d) // math.factorial(d-r)
        Pr = S * assignments / (d ** k)
        probabilities.append(Pr)
    return probabilities

def calculate_probabilities(n, m):
    '''

    :param n: quantity of numbers by set
    :param m: number of digits after the decimal point
    :return:
    '''

    d = 10

    # Probabilités des différentes configurations
    P_poker, P_full_carre, P_brelan_double_paire, P_paire, P_rien = poker_probabilities(m, d)

    prob = {
        "Poker": P_poker,
        "Carré ou Full": P_full_carre,
        "Brelan ou Double Paire": P_brelan_double_paire,
        "Paire": P_paire,
        "Rien": P_rien
    }

    return {k: p * n for k, p in prob.items()}

#calculer la proba d'avoir n case differente
def poker_test(sequence):
    '''

    :param sequence: list of numbers
    :param m: number of digits after the decimal point
    :return:
    '''
    if(type(sequence[0]) == float):
        for i in range(len(sequence)):
            sequence[i] = format(sequence[i], '.5f')
    # Calculate the observed counts
    m = len(sequence[0][2:])
    categories = [poker_category(digits[2:]) for digits in sequence]
    observed = Counter(categories)

    observed_counts = {cat: observed[cat] for cat in observed}

    # Calculate the expected probabilities
    expected_counts = calculate_probabilities(len(sequence), m)

    # Associer les probabilités aux catégories
    categories = ["Poker", "Carré ou Full", "Brelan ou Double Paire", "Paire", "Rien"]

    # Combinaison des comptes Carré et Full
    observed_counts['Carré ou Full'] = observed_counts.pop('Carré') + observed_counts.pop('Full')

    # Combinaison des comptes Brelan et Double Paire
    observed_counts['Brelan ou Double Paire'] = observed_counts.pop('Brelan') + observed_counts.pop('Double paire')

    observed_counts = dict(sorted(observed_counts.items()))
    expected_counts = dict(sorted(expected_counts.items()))


    chi2_stat = sum((observed_counts.get(cat, 0) - expected_counts.get(cat, 0)) ** 2 / expected_counts.get(cat, 0) for cat in expected_counts)
    df = len(expected_counts) - 1
    chi2_critical = chi2.ppf(1 - 0.001, df)
    reject_null = chi2_stat > chi2_critical

    '''
    print(f"Observed counts: {observed_counts}")
    print(f"Expected counts: {expected_counts}")
    print(f"Chi-square statistic: {chi2_stat}")
    print(f"Critical value: {chi2_critical}")
    print(f"Reject null hypothesis: {reject_null}")
    print(f"Uniform generator : {chi2_stat < chi2_critical}")
    '''
    return {
        "observed_counts": observed_counts,
        "expected_counts": expected_counts,
        "chi_square_statistic": chi2_stat,
        "critical_value": chi2_critical,
        "reject_null": reject_null,
        "Uniform generator": chi2_stat < chi2_critical
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
    degrees_of_freedom = len(expected_counts) - 1

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
    #result = runs_test([next(random_number_generator) for _ in range(1000)])

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
    result = gap_test(random_numbers, 0, 0.5, 5, 1000)

    with open("python_generator_test_results.txt", "w") as f:
        f.write(f"----------- Gap test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")

    # Perform a poker test on the generated random numbers
    # The poker test checks if the numbers in the sequence are randomly distributed
    result = poker_test(random_numbers)

    with open("python_generator_test_results.txt", "a") as f:
        f.write(f"----------- Poker test -----------\n")
        f.write(f"Observed Counts: {result['observed_counts']}\n")
        f.write(f"Expected Counts: {result['expected_counts']}\n")
        f.write(f"Chi2 Statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical Value: {result['critical_value']}\n")
        f.write(f"Reject Null Hypothesis: {result['reject_null']}\n")
        f.write(f"Uniform generator : {result['Uniform generator']}\n")

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

    #custom_generator_test(pi_decimals)

    python_generator_test()

    #pi_decimals_test(pi_decimals)

    # Perform a poker test on the decimals of pi
    #nombres = [random.uniform(0, 1) for _ in range(1000)]
    #for i in range(len(nombres)):
    #    nombres[i] = format(nombres[i], '.5f')
    #poker_test(nombres)



