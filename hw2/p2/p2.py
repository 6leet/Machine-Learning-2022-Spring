import math
import sys

from sympy import beta

def preprocess(filename):
    trails = []
    with open(filename) as f:
        trail = f.readline()[:-1]
        while trail:
            trails.append(trail)
            trail = f.readline()[:-1]

    return trails

def c(n, m):
    return math.factorial(n) / (math.factorial(n - m) * math.factorial(m))

def binomial_likelihood(m, p, N):
    return c(N, m) * pow(p, m) * pow(1 - p, N - m)

def beta_binomial_conjugation(trails, a, b):
    for i, trail in zip(range(len(trails)), trails):
        N = len(trail)
        head = sum([int(t) for t in trail])
        likelihood = binomial_likelihood(head, head / N, N)
        print('case ' + str(i + 1) + ':', trail)
        print('Likelihood:', likelihood)
        print('Beta prior:    ', 'a = ', a, 'b = ', b)
        print('Beta posterior:', 'a = ', a + head, 'b = ', b + N - head)
        print()
        a += head
        b += N - head

if len(sys.argv) < 4:
    print('Usage: python3 p2.py <txt_file> <a> <b>')

filename, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
trails = preprocess(filename)
beta_binomial_conjugation(trails, a, b)