import math
import random
class Generator:
    def __init__(self, seed=None, pi_decimals=None):
        if pi_decimals is None:
            self.pi_decimals = math.pi
            self.pi_decimals = str(self.pi_decimals)[2:]
            self.pi_decimals = [int(digit) for digit in self.pi_decimals]
        else:
            self.pi_decimals = pi_decimals
        self.pi_index = 0
        self.modulus = 2**32
        self.multiplier = 1103515245
        self.increment = 12345
        if seed is not None:
            self.seed = seed


    def seed(self, seed):
        self.pi_index = seed % len(self.pi_decimals)

    def random(self):
        print(f"pi_index: {self.pi_index}")
        x = self.pi_decimals[self.pi_index]
        x = (x * self.multiplier + self.increment) % self.modulus
        self.pi_index = random.randint(0, len(self.pi_decimals) - 1)
        return x / self.modulus

    def uniform(self, a=0, b=1):
        return a + (b - a) * self.random()




def get_pi_decimals(file):
    with open(file, 'r') as f:
        pi_decimals = f.read().strip()
    return pi_decimals

decimals = get_pi_decimals("pi_decimals_single_line.txt")

generator = Generator(pi_decimals=[int(digit) for digit in decimals])

for _ in range(100):
    print(generator.uniform(0, 1))