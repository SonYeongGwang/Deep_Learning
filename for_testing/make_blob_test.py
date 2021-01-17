from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

(x, y) = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0, random_state=1)
(x2, y2) = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.5, random_state=1)

print(x, np.shape(x))
print(y, np.shape(y))

y = y.reshape((y.shape[0], 1))
x = np.c_[x, np.ones(x.shape[0])]

print(np.shape(y), np.shape(x))

plt.style.use("ggplot")
plt.figure()
plt.title("x, y")
plt.scatter(x[:,0], x[:,1], c=y)

plt.style.use("ggplot")
plt.figure()
plt.title("x2, y2")
plt.scatter(x2[:,0], x2[:,1], c=y2)
plt.show()