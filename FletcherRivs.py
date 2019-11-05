import numpy as np
from numpy import linalg as LA

INF = 1e9 + 7
eps = 1e-6
eps2 = 1e-6

def func1(x, a, b, c, d):
    return 100 * (c + x * d - a ** 2 - 2 * a * b * x - (x * b) ** 2) ** 2 + 5 * (1 - a - x * b) ** 2

def ternarny_search(my_func):
    lt = 0
    r = 30
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


# my_func = lambda x: func(x, 3, 4, 5, 6)

# a = [int(x) for x in input().split()]
# a = np.array(a)
# print(type(a))
# print(a)
func = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + 5 * (1 - x[0]) ** 2
grad = lambda x: np.array([-400 * x[0] * x[1] + 400 * x[0] ** 3 + 10 * x[0] - 10, 200 * x[1] - 200 * x[0] ** 2])

# print(fucking_func(a))
# print(fucking_grad(a))
# print(-1 * a)
# print(fucking_grad(-1 * a))
# # print("%.7f" % float(eps))


X_prev = np.array([5, 7])
S_prev = -1 * grad(X_prev)
n = 10
k = j = 0

while True:
    my_func = lambda x: func1(x, X_prev[0], S_prev[0], X_prev[1], S_prev[1])
    k = ternarny_search(my_func)
    X = X_prev + k * S_prev
    w = (LA.norm(grad(X)) / LA.norm(grad(X_prev))) ** 2
    S = -1 * grad(X) + w * S_prev
    # print(X)
    if LA.norm(S) < eps2 or LA.norm(X - X_prev) < eps2:
        print(X)
        break
    elif j + 1 < n:
        j = j + 1
    else:
        j = 0
        k = k + 1
        X_prev = X
        S_prev = -1 * grad(X_prev)
