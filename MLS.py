import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

#Constants
NODES = 5
DEGREE = 3

#Initial and polinom getter functions respectively
func = lambda x: x ** 2 + np.sin(x)
P = lambda x, a: sum([a[i] * float(x ** i) for i in range(0, DEGREE + 1)])

#LEAST squares method begin
x = np.array([random.uniform(-1, 1) for i in range(NODES)])
Q = np.full((NODES, NODES), 0, dtype=float)
for i in range(0, NODES):
    for j in range(0, DEGREE + 1):
        Q[i][j] = x[i] ** j
H = Q.T.dot(Q)
y = np.fromiter(map(func, x), float)
b = Q.T.dot(y)
#Dodging from singular case
for i in range(0, NODES):
    for j in range(DEGREE + 1, NODES):
        H[i][j] = i

solve = LA.solve(H, b)
#end


#Lezhandr method
#Definition tbe multipliers of searchable polinom
cn = np.array([1 / 3, (np.sin(1) - np.cos(1)) * 3, 2 / 3, (28*np.cos(1) - 18 * np.sin(1)) / (2 / 7)])
#endjkh


#Initial data
a = -1
b = 1
m = 100

t = np.arange(a, b + (b - a) / m, (b - a) / m)
s = func(t)
plt.subplot(3, 1, 1)
plt.plot(t, s, '-', lw=2)
plt.title('Plot the legit graph of function')
plt.grid(True)

s = [P(x, solve) for x in t]
plt.subplot(3, 1, 2)
plt.plot(t, s, '-', lw=2)
plt.title('Plot the graph of function based on least squares method')
plt.grid(True)

s = [P(x, cn) for x in t]
plt.subplot(3, 1, 3)
plt.plot(t, s, '-', lw=2)
plt.title('Plot the graph of function based on Lezhandr\'s polynom')
plt.grid(True)

plt.show()

s = [func(x) - P(x, cn) for x in t]
plt.subplot(2, 1, 1)
plt.plot(t, s, '-', lw=2)
plt.title('Substraction plot between the initial function and generalized polynom')
plt.grid(True)


s = [func(x) - P(x, solve) for x in t]
plt.subplot(2, 1, 2)
plt.plot(t, s, '-', lw=2)
plt.title('Substraction plot between the initial function and Lezhandr\'s polynom')
plt.grid(True)

plt.show()

#Computing the total error
integral = 7/5 - np.sin(2) / 2
q2 = [2, 2/3, 2/5, 2/7]
qn = integral - sum([(cn[i] ** 2) * q2[i] for i in range(0, 4)])
print(qn)