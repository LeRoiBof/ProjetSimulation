import random

def generate_random_numbers(pi_decimals, slice_size):
    while True:
        index = random.randint(0, len(pi_decimals) - 1)
        if index + slice_size > len(pi_decimals):
            index = 0
        slice = pi_decimals[index:index + slice_size]
        random_number = int(slice)

        yield random_number / 10 ** slice_size


def basic_random_generator(pi_decimals):
    while True:
        index = random.randint(0, len(pi_decimals) - 1)
        random_number = int(pi_decimals[index])
        yield random_number / 10


def get_pi_decimals(file):
    with open(file, 'r') as f:
        pi_decimals = f.read().strip()
    return pi_decimals

decimals = get_pi_decimals("pi_decimals_single_line.txt")


random_number_generator = generate_random_numbers(decimals, 5)
basic_random_number = basic_random_generator(decimals)

for _ in range(100):
    print(next(random_number_generator), next(basic_random_number))

# for _ in range(100):
#     print(next(random_number_generator))