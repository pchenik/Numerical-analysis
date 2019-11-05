import numpy as np
from numpy import linalg as LA
import random

INF = 1e9 + 7
eps = 1e-5
eps2 = 1e-5

def func1(x, a, b, c, d):
    return ((a + x * b) ** 2 + c + d * x - 11) ** 2 + (a + b * x + (c + x * d) ** 2 - 7) ** 2

def ternarny_search(my_func):
    lt = 0
    r = 100
    while r - lt > eps:
        lm = lt + (r - lt) / 3
        rm = r - (r - lt) / 3
        hl = my_func(lm)
        hr = my_func(rm)
        if hl < hr:
            r = rm
        else:
            lt = lm
    return (lt + r) / 2


func = lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
grad = lambda x: np.array([4 * x[0] ** 3 + 4 * x[0] * x[1] - 42 * x[0] + 2 * x[1] ** 2 - 14,
                                    2 * x[0] ** 2 + 4 * x[0] * x[1] + 4 * x[1] ** 3 - 26 * x[1] - 22])

for i in range(10):
    X_rand = random.randrange(-10, 10)
    Y_rand = random.randrange(-10, 10)
    X_prev = np.array([X_rand, Y_rand])
    S_prev = -1 * grad(X_prev)
    print(X_prev)
    while True:
        kek = grad(X_prev)
        my_func = lambda x: func1(x, X_prev[0], -kek[0], X_prev[1], -kek[1])
        k = ternarny_search(my_func)
        X = X_prev - k * kek
        # print(X)
        if LA.norm(grad(X)) < eps2 or LA.norm(X - X_prev) < eps2:
            print(X)
            break
        else:
            X_prev = X
