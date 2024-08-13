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


def generate_lcg(seed, a=1664525, c=1013904223, m=2 ** 32):
    state = seed
    while True:
        state = (a * state + c) % m
        yield state / m


def pi_based_lcg(pi_decimals, num_values):
    # Extraire un entier comme graine à partir des décimales de π
    seed = int(pi_decimals[:10])  # Utiliser les 10 premières décimales pour la graine
    lcg = generate_lcg(seed)

    # Produire num_values valeurs
    for _ in range(num_values):
        yield next(lcg)

# This function generates random numbers based on the decimals of pi.
# It yields a new random number each time it is called.
def generate_uniform_random_numbers(pi_decimals):
    # Extract an integer as a seed from the decimals of pi
    seed = int(pi_decimals[:10])  # Use the first 10 decimals for the seed

    # Generate random numbers using a linear congruential generator
    lcg = generate_lcg(seed)

    # Generate an infinite sequence of random numbers
    while True:
        yield next(lcg)


def gap_test(sequence, alpha):
    sequence_length = len(sequence)
    m = 5

    # Intervalle [a, b]
    a, b = 0.1, 0.5  # Travailler avec des valeurs normalisées entre 0 et 1
    p = (b - a)  # Probabilité p de tomber dans [a, b]

    # Conversion des valeurs dans l'intervalle [0, 1]
    sequence = [float(f"0.{str(x)[2:]}") for x in sequence]

    # Marquage des valeurs dans l'intervalle [a, b]
    marqués = np.zeros(sequence_length, dtype=bool)
    for i in range(sequence_length):
        if a <= sequence[i] <= b:
            marqués[i] = True

    # Calcul des gaps
    gaps = []
    gap = 0
    for marqué in marqués:
        if marqué:
            gaps.append(gap)
            gap = 0
        else:
            gap += 1
    gaps.append(gap)  # Ajouter le dernier gap

    # Calcul des probabilités théoriques des gaps
    max_gap = max(gaps)
    probs_théoriques = [(1 - p) ** i * p for i in range(max_gap + 1)]
    probs_théoriques.append((1 - p) ** (max_gap + 1))

    # Calcul des occurrences observées
    occurrences = np.bincount(gaps)

    # Élargir la liste des occurrences si nécessaire
    if len(occurrences) < len(probs_théoriques):
        occurrences = np.append(occurrences, [0] * (len(probs_théoriques) - len(occurrences)))

    # Nombre de gaps
    N = len(gaps)

    # Valeurs attendues
    valeurs_attendues = [N * p for p in probs_théoriques]

    # Test du chi²
    chi2_stat = np.sum((occurrences - valeurs_attendues) ** 2 / valeurs_attendues)
    chi2_critical = chi2.ppf(1 - alpha, len(probs_théoriques) - 1)

    reject_null = chi2_stat > chi2_critical

    return {
        "observed_counts": occurrences,
        "expected_counts": valeurs_attendues,
        "chi_square_statistic": chi2_stat,
        "critical_value": chi2_critical,
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

def stirling_number(n, k):
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
def poker_test(sequence, alpha):
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
    if 'Carré' in observed_counts and 'Full' in observed_counts:
        observed_counts['Carré ou Full'] = observed_counts.pop('Carré') + observed_counts.pop('Full')
    else:
        observed_counts['Carré ou Full'] = 0
        if 'Carré' in observed_counts:
            observed_counts['Carré ou Full'] += observed_counts.pop('Carré')
        if 'Full' in observed_counts:
            observed_counts['Carré ou Full'] += observed_counts.pop('Full')

    # Combinaison des comptes Brelan et Double Paire
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
    intervals = np.linspace(0, 1, 11)

    observed_frequencies, _ = np.histogram(sequence, bins=intervals)

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
    result = gap_test(random_number,alpha)
    # a = 0.1
    # b = 0.5
    # m = 5

    # Write the result to a file
    with open("custom_generator_test_results.txt", "w") as f:
        f.write(f"----------- Gap test ----eae-------\n")
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
        f.write(f"Uniform generator : {result['Uniform generator']}\n")

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
    result = chi2_test(random_numbers,alpha)

    with open("python_generator_test_results.txt", "a") as f:
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

    #pi_decimals_test(pi_decimals)



