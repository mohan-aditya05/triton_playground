from numba import njit
import random
import time

@njit
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

start = time.time()
monte_carlo_pi(100000)
print("Time taken: ", time.time() - start)