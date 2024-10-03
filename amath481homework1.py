import numpy as np

func = lambda x : x*np.sin(3*x)-np.exp(x)
funcderiv = lambda x : np.sin(3*x) + (3*x*np.cos(3*x)) - np.exp(x)
A1 = np.array([-1.6])
while abs(func(A1[-1])) > 1e-6:
    A1 = np.append(A1, A1[-1] - (func(A1[-1]))/(funcderiv(A1[-1])))
A1 = np.append(A1, A1[-1] - (func(A1[-1]))/(funcderiv(A1[-1]))) # since Gradescope checks one additional step

lowerbound = -0.7
upperbound = -0.4
A2 = []
while True:
    midpt = (lowerbound+upperbound)/2
    A2.append(midpt)
    funcval = func(midpt)
    if funcval>0:
        lowerbound=midpt
    else:
        upperbound=midpt
    if abs(funcval) < 1e-6:
        break
A2 = np.array(A2)

A3 = np.array([len(A1)-1, len(A2)])

A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])
A4 = A+B
A5 = 3*x - 4*y
A6 = A@x
A7 = B@(x-y)
A8 = D@x
A9 = D@y + z
A10 = A@B
A11 = B@C
A12 = C@D