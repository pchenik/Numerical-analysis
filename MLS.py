import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

INF = 1e9 + 7
eps = 1e-6
eps2 = 1e-4


func = lambda x: x ** 2 + np.sin(x)
P = lambda x, a: sum([a[i] * float(x ** i) for i in range(0, 4)])

#LEAT squares method begin
x = np.array([-0.9, -0.4, -0.3, 0.5, 0.7])
Q = np.full((5, 5), 0, dtype=float)
for i in range(0, 5):
    for j in range(0, 4):
        Q[i][j] = x[i] ** j
#print(Q)
H = Q.T.dot(Q)
y = np.fromiter(map(func, x), float)
b = Q.T.dot(y)
for i in range(0, 5):
    Q[i][4] = i
solve = LA.solve(Q, b)
#end


#Lezhandr method:
cn = np.array([1 / 3, (np.sin(1) - np.cos(1)) / 3, 2 / 3, (28*np.cos(1) - 18 * np.sin(1)) / (2 / 7)])
#end



a = -1
b = 1
m = 100

t = np.arange(a, b + (b - a) / m, (b - a) / m)
s = func(t)
plt.subplot(3, 1, 1)
plt.plot(t, s, '-', lw=2)
plt.title('Plot the legit graph of function')
plt.grid(True)

s = [P(x, solve) for x in t];
plt.subplot(3, 1, 2)
plt.plot(t, s, '-', lw=2)
plt.title('Plot the graph of function based on least squares method')
plt.grid(True)

s = [P(x, cn) for x in t]
plt.subplot(3, 1, 3)
plt.plot(t, s, '-', lw=2)
plt.title('Plot the graph of function based on Lezhandr\'s polinom')
plt.grid(True)

plt.show()