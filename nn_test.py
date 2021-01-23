import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([[1], [3], [5]])

def add(x, y):
    return x[0] + y

for (x, target) in zip(X, y):
    print(x, target)
    pred = add(x, target)[0][0]
    print(pred)