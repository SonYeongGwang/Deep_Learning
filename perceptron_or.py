from ImageTools.pyimagesearch import Perceptron
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [1]])

pr = Perceptron(X.shape[1])
pr.fit(X, y, epochs=20)

print("[INFO] testing perceptron...")

for (x, target) in zip(X, y):
    pred = pr.predict(x)
    print("[INFO] data={}, ground-truth]{}, prediction={}".format(x, target[0], pred))

plt.scatter(X[:,0], X[:,1], marker='o', c=y, s=30)
plt.show()