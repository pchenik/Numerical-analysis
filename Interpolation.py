import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

INF = 1e9 + 7
eps = 1e-6
eps2 = 1e-4
Pi = np.arccos(-1)

def L(func, xn, x):
    res = 0
    for i in range(0, len(xn)):
        cur = 1
        for j in range(0, len(xn)):
            if i != j:
                cur *= (x - xn[j]) / (xn[i] - xn[j])
        res += cur * func(xn[i])
    return res

def solve(func):

    a = -1
    b = 1
    m = 200
    n = 10


    t = np.arange(a, b + (b - a) / m, (b - a) / m)
    s = func(t)
    plt.subplot(3, 1, 1)
    plt.plot(t, s, '-', lw=2)
    plt.title('Plot the legit graph of function')
    plt.grid(True)

    #xn = np.arange(a, b + (b - a) / n, (b - a) / n)
    #print(*xn)
    #s = [L(func, xn, x) for x in t]
    #plt.subplot(3, 1, 2)
    #plt.plot(t, s, '-', lw=2)
    #plt.title('Plot by Lagrange method with equally distributing points')
    #plt.grid(True)

    xn = [0.5 * ((b - a) * np.cos((2 * i + 1) / (2 * n + 2) * Pi) + (b + a)) for i in range(0, n)]
    #print(len(xn))
    print(*xn)
    s = [L(func, xn, x) for x in t]
    #s = [L(func, xn, x) - func(x) for x in t]
    plt.subplot(3, 1, 3)
    plt.plot(t, s, '-', lw=2)
    #plt.title('Plot by Lagrange method with Chebyshev distribution')
    plt.grid(True)

    s = [L(func, xn, x) - func(x) for x in t]
    plt.subplot(3, 1, 2)
    plt.plot(t, s, '-', lw=2)
    #plt.title('Plot by Lagrange method with Chebyshev distribution')
    plt.grid(True)


    plt.show()

f = lambda x: x ** 2 * 4 * np.sin(x) - 2
h = lambda x : abs(x) * f(x)

solve(f)
#print('\nAnother funtion:\n')
#solve(h)