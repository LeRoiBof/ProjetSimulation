import numpy as np
from scipy import stats

decimales_pi = np.loadtxt('pi_decimals.txt', dtype=str)

k = 10
observed_frequencies, _ = np.histogram(decimales_pi, bins=k)
expected_frequencies = np.ones(k) * len(decimales_pi) / k
chi2, p_value_chi2 = stats.chisquare(observed_frequencies, expected_frequencies)

d, p_value_ks = stats.kstest(decimales_pi, 'uniform')

print(f"Test du x2 : chi2 = {chi2}, p-value = {p_value_chi2}")
print(f"Test de Kolmogorov-Smirnov : D = {d}, p-value = {p_value_ks}")