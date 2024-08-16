import random
from collections import Counter
import math
import numpy as np
from scipy.stats import chi2

def get_pi_decimals(file):
    """
    Read the decimals of pi from a file.

    Parameters:
    file (str): The path to the file containing the decimals of pi.

    Returns:
    str: A string containing the decimals of pi.
    """
    with open(file, 'r') as f:
        pi_decimals = f.read().strip()
    return pi_decimals


def generate_lcg(seed, a=1664525, c=1013904223, m=2 ** 32):
    """
    Generate a sequence of random numbers using a Linear Congruential Generator (LCG).

    The LCG is a simple pseudo-random number generator algorithm that uses a linear equation to generate a sequence of numbers.

    Parameters:
    seed (int): The initial value (seed) for the generator.
    a (int): The multiplier coefficient. Default is 1664525.
    c (int): The increment coefficient. Default is 1013904223.
    m (int): The modulus. Default is 2^32.

    Yields:
    float: A pseudo-random number in the range [0, 1).
    """
    state = seed
    while True:
        state = (a * state + c) % m
        yield state / m


def generate_uniform_random_numbers(pi_decimals):
    """
    Generate an infinite sequence of uniform random numbers using a Linear Congruential Generator (LCG).

    This function extracts a seed from the decimals of pi and uses it to initialize the LCG.

    Parameters:
    pi_decimals (str): A string containing the decimals of pi.

    Yields:
    float: A pseudo-random number in the range [0, 1).
    """
    # Extract an integer as a seed from the decimals of pi
    start_index = random.randint(0, len(pi_decimals) - 10)
    seed = int(pi_decimals[start_index:start_index + 10])

    # Generate random numbers using a linear congruential generator
    lcg = generate_lcg(seed)

    # Generate an infinite sequence of random numbers
    while True:
        yield next(lcg)


def gap_test(sequence, alpha):
    """
    Perform a gap test on a sequence of numbers to check for uniform distribution.

    The gap test checks the intervals (gaps) between occurrences of numbers within a specified range [a, b].
    It then compares the observed gap frequencies with the expected frequencies using a chi-square test.

    Parameters:
    sequence (list of float): The sequence of numbers to be tested.
    alpha (float): The significance level for the chi-square test.

    Returns:
    dict: A dictionary containing the observed counts, expected counts, chi-square statistic, critical value, and a boolean indicating whether to reject the null hypothesis.
    """
    sequence_length = len(sequence)
    m = 5

    # Interval [a, b]
    a, b = 0.2, 0.8  # Working with normalized values between 0 and 1
    p = (b - a)  # Probability p of falling within [a, b]

    # Convert values to the interval [0, 1]
    sequence = [float(f"0.{str(x)[2:]}") for x in sequence]

    # Mark values within the interval [a, b]
    marked = np.zeros(sequence_length, dtype=bool)
    for i in range(sequence_length):
        if a <= sequence[i] <= b:
            marked[i] = True

    # Calculate gaps
    gaps = []
    gap = 0
    for mark in marked:
        if mark:
            gaps.append(gap)
            gap = 0
        else:
            gap += 1
    gaps.append(gap)  # Add the last gap

    # Calculate theoretical probabilities of gaps
    max_gap = max(gaps)
    theoretical_probs = [(1 - p) ** i * p for i in range(max_gap + 1)]
    theoretical_probs.append((1 - p) ** (max_gap + 1))

    # Calculate observed occurrences
    occurrences = np.bincount(gaps)

    # Extend the list of occurrences if necessary
    if len(occurrences) < len(theoretical_probs):
        occurrences = np.append(occurrences, [0] * (len(theoretical_probs) - len(occurrences)))

    # Number of gaps
    N = len(gaps)

    # Expected values
    expected_values = [N * p for p in theoretical_probs]

    # Chi-square test
    chi2_stat = np.sum((occurrences - expected_values) ** 2 / expected_values)

    chi2_critical = chi2.ppf(1 - alpha, len(theoretical_probs) - 1)

    reject_null = chi2_stat > chi2_critical

    return {
        "observed_counts": occurrences,
        "expected_counts": expected_values,
        "chi_square_statistic": chi2_stat,
        "critical_value": chi2_critical,
        "reject_null": reject_null
    }

# Fonction pour déterminer la catégorie de poker
def poker_category(rolls):
    """
    Determine the poker category of a given set of rolls.

    This function categorizes a set of rolls into one of the poker hand categories such as Poker, Carré, Full, Brelan, Double paire, Paire, or Rien.

    Parameters:
    rolls (list of int): A list of integers representing the rolls.

    Returns:
    str: The category of the poker hand.
    """
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

def stirling_number(n, k):
    """
    Calculate the Stirling number of the second kind.

    The Stirling number of the second kind, S(n, k), represents the number of ways to partition a set of n objects into k non-empty subsets.

    Parameters:
    n (int): The total number of objects.
    k (int): The number of non-empty subsets.

    Returns:
    int: The Stirling number of the second kind, S(n, k).
    """
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    return k * stirling_number(n-1, k) + stirling_number(n-1, k-1)

def poker_probabilities(k, d):
    """
    Calculate the probabilities of different poker hand categories.

    This function calculates the probabilities of obtaining different poker hand categories given the number of rolls and the number of distinct values.

    Parameters:
    k (int): The number of rolls.
    d (int): The number of distinct values.

    Returns:
    list of float: A list of probabilities for each poker hand category.
    """
    probabilities = []
    for r in range(1, k+1):
        S = stirling_number(k, r)
        assignments = math.factorial(d) // math.factorial(d-r)
        Pr = S * assignments / (d ** k)
        probabilities.append(Pr)
    return probabilities

def calculate_probabilities(n, m):
    """
    Calculate the probabilities of different poker hand categories for a given set size and number of digits.

    This function calculates the probabilities of obtaining different poker hand categories given the number of numbers in a set and the number of digits after the decimal point.

    Parameters:
    n (int): The quantity of numbers by set.
    m (int): The number of digits after the decimal point.

    Returns:
    dict: A dictionary containing the probabilities for each poker hand category.
    """
    d = 10

    # Probabilities of different configurations
    P_poker, P_full_carre, P_brelan_double_paire, P_paire, P_rien = poker_probabilities(m, d)

    prob = {
        "Poker": P_poker,
        "Carré ou Full": P_full_carre,
        "Brelan ou Double Paire": P_brelan_double_paire,
        "Paire": P_paire,
        "Rien": P_rien
    }

    return {k: p * n for k, p in prob.items()}

def poker_test(numbers, alpha):
    """
    Perform a poker test on a sequence of numbers to check for randomness.

    The poker test categorizes the numbers into poker hand categories and compares the observed frequencies with the expected frequencies using a chi-square test.

    Parameters:
    sequence (list of float): The sequence of numbers to be tested.
    alpha (float): The significance level for the chi-square test.

    Returns:
    dict: A dictionary containing the observed counts, expected counts, chi-square statistic, critical value, and a boolean indicating whether to reject the null hypothesis.
    """
    sequence = numbers.copy()
    if isinstance(sequence[0], float):
        for i in range(len(sequence)):
            sequence[i] = str(sequence[i])

    sequence_new = []

    for i in range(len(sequence)):
        if(sequence[i][1] == "."):
            sequence[i] = sequence[i][2:]

    for i in range(len(sequence)):
        for j in range(0, len(sequence[i])-5, 5):
            sequence_new.append(sequence[i][j: j+5])

    if sequence_new != [] :
        sequence = sequence_new

    for i in range(len(sequence)):
        sequence[i] = "0." + sequence[i]

    #print(sequence)
    # Calculate the observed counts
    m = len(sequence[0][2:])

    #print(sequence)
    categories = [poker_category(digits[2:]) for digits in sequence]
    observed = Counter(categories)

    observed_counts = {cat: observed[cat] for cat in observed}

    # Calculate the expected probabilities
    expected_counts = calculate_probabilities(len(sequence), m)

    # Associate the probabilities with categories
    categories = ["Poker", "Carré ou Full", "Brelan ou Double Paire", "Paire", "Rien"]

    # Combine counts for Carré and Full
    if 'Carré' in observed_counts and 'Full' in observed_counts:
        observed_counts['Carré ou Full'] = observed_counts.pop('Carré') + observed_counts.pop('Full')
    else:
        observed_counts['Carré ou Full'] = 0
        if 'Carré' in observed_counts:
            observed_counts['Carré ou Full'] += observed_counts.pop('Carré')
        if 'Full' in observed_counts:
            observed_counts['Carré ou Full'] += observed_counts.pop('Full')

    # Combine counts for Brelan and Double Paire
    if 'Brelan' in observed_counts and 'Double paire' in observed_counts:
        observed_counts['Brelan ou Double Paire'] = observed_counts.pop('Brelan') + observed_counts.pop('Double paire')
    else:
        observed_counts['Brelan ou Double Paire'] = 0
        if 'Brelan' in observed_counts:
            observed_counts['Brelan ou Double Paire'] += observed_counts.pop('Brelan')
        if 'Double paire' in observed_counts:
            observed_counts['Brelan ou Double Paire'] += observed_counts.pop('Double paire')

    observed_counts = dict(sorted(observed_counts.items()))
    expected_counts = dict(sorted(expected_counts.items()))

    chi2_stat = sum((observed_counts.get(cat, 0) - expected_counts.get(cat, 0)) ** 2 / expected_counts.get(cat, 0) for cat in expected_counts)
    df = len(expected_counts) - 1
    chi2_critical = chi2.ppf(1 - alpha, df)
    reject_null = chi2_stat > chi2_critical

    return {
        "observed_counts": observed_counts,
        "expected_counts": expected_counts,
        "chi_square_statistic": chi2_stat,
        "critical_value": chi2_critical,
        "reject_null": reject_null,
        "Uniform generator": chi2_stat < chi2_critical
    }

def chi2_test(sequence, alpha):
    """
    Perform a chi-square test on a sequence of numbers to check for uniform distribution.

    The chi-square test compares the observed frequencies of numbers in different intervals with the expected frequencies.

    Parameters:
    sequence (list of float): The sequence of numbers to be tested.
    alpha (float): The significance level for the chi-square test.

    Returns:
    dict: A dictionary containing the observed counts, expected counts, chi-square statistic, critical value, and a boolean indicating whether to reject the null hypothesis.
    """
    intervals = np.linspace(0, 1, 11)

    observed_frequencies, _ = np.histogram(sequence, bins=intervals)

    print(observed_frequencies)

    expected_frequencies = np.ones(len(observed_frequencies)) * len(sequence) / len(observed_frequencies)

    chi2_stat = np.sum((observed_frequencies - expected_frequencies) ** 2 / expected_frequencies)

    df = len(observed_frequencies) - 1

    chi2_critical = chi2.ppf(1 - alpha, df)

    reject_null = chi2_stat > chi2_critical

    return {
        "observed_counts": observed_frequencies,
        "expected_counts": expected_frequencies,
        "chi_square_statistic": chi2_stat,
        "critical_value": chi2_critical,
        "reject_null": reject_null
    }

def custom_generator_test(pi_decimals, sequence_length, alpha):
    # Generate random numbers based on the decimals of pi
    random_number_generator = generate_uniform_random_numbers(pi_decimals)

    random_number = [next(random_number_generator) for _ in range(sequence_length)]

    # Perform a gap test on the generated random numbers
    # The gap test checks if the numbers in the sequence are uniformly distributed
    result = gap_test(random_number, alpha)
    # a = 0.1
    # b = 0.5
    # m = 5

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
    result = poker_test(random_number, alpha)

    with open("custom_generator_test_results.txt", "a") as f:
        f.write(f"----------- Poker test -----------\n")
        f.write(f"Observed Counts: {result['observed_counts']}\n")
        f.write(f"Expected Counts: {result['expected_counts']}\n")
        f.write(f"Chi2 Statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical Value: {result['critical_value']}\n")
        f.write(f"Reject Null Hypothesis: {result['reject_null']}\n")

    # Perform a Chi-Square test on the generated random numbers
    # The Chi-Square test checks if the observed counts of numbers in the sequence are significantly different from the expected counts
    result = chi2_test(random_number, alpha)

    with open("custom_generator_test_results.txt", "a") as f:
        f.write(f"----------- Chi-Square test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")


def python_generator_test(sequence_length, alpha):
    # Generate random numbers using the Python random module
    random_numbers = [random.uniform(0, 1) for _ in range(sequence_length)]

    # Perform a gap test on the generated random numbers
    # The gap test checks if the numbers in the sequence are uniformly distributed
    result = gap_test(random_numbers, alpha)
    with open("python_generator_test_results.txt", "w") as f:
        f.write(f"----------- Gap test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")

    # Perform a poker test on the generated random numbers
    # The poker test checks if the numbers in the sequence are randomly distributed
    result = poker_test(random_numbers, alpha)

    with open("python_generator_test_results.txt", "a") as f:
        f.write(f"----------- Poker test -----------\n")
        f.write(f"Observed Counts: {result['observed_counts']}\n")
        f.write(f"Expected Counts: {result['expected_counts']}\n")
        f.write(f"Chi2 Statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical Value: {result['critical_value']}\n")
        f.write(f"Reject Null Hypothesis: {result['reject_null']}\n")

    # Perform a Chi-Square test on the generated random numbers
    # The Chi-Square test checks if the observed counts of numbers in the sequence are significantly different from the expected counts
    print(random_numbers[0])
    result = chi2_test(random_numbers, alpha)

    with open("python_generator_test_results.txt", "a") as f:
        f.write(f"----------- Chi-Square test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")

def pi_decimal_test(pi_decimal, alpha):
    random_numbers = []
    for i in range(0,len(pi_decimal) - 5, 5):
        random_numbers.append("0."+pi_decimal[i:(i+5)])
    # Perform a gap test
    # The gap test checks if the numbers in the sequence are uniformly distributed
    result = gap_test(random_numbers, alpha)
    with open("pi_decimals_test_results.txt", "w") as f:
        f.write(f"----------- Gap test -----------\n")
        f.write(f"Observed counts: {result['observed_counts']}\n")
        f.write(f"Expected counts: {result['expected_counts']}\n")
        f.write(f"Chi-square statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical value: {result['critical_value']}\n")
        f.write(f"Reject null hypothesis: {result['reject_null']}\n")

    # Perform a poker test on the generated random numbers
    # The poker test checks if the numbers in the sequence are randomly distributed
    result = poker_test(random_numbers, alpha)

    with open("pi_decimals_test_results.txt", "a") as f:
        f.write(f"----------- Poker test -----------\n")
        f.write(f"Observed Counts: {result['observed_counts']}\n")
        f.write(f"Expected Counts: {result['expected_counts']}\n")
        f.write(f"Chi2 Statistic: {result['chi_square_statistic']}\n")
        f.write(f"Critical Value: {result['critical_value']}\n")
        f.write(f"Reject Null Hypothesis: {result['reject_null']}\n")

    # Perform a Chi-Square test on the generated random numbers
    # The Chi-Square test checks if the observed counts of numbers in the sequence are significantly different from the expected counts
    result = chi2_test(random_numbers, alpha)

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


    alpha = 0.05
    sequence_length = 1000

    custom_generator_test(pi_decimals, sequence_length, alpha)

    python_generator_test(sequence_length, alpha)

    pi_decimal_test(pi_decimals, alpha)



