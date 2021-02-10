import numpy as np
import matplotlib.pyplot as plt
import math

# a = 6
# b = 3
# c = 2.0

# print(a + b)
# print(a + c)

# A = [1, 2, 3]
# B = [1, 2]

# for a, b in zip(A, B):
#     print(a, b)


# ## cross-entropy cost function
# y = []
# for x in np.arange(0.0009, 1, step=0.001):
#     y.append(math.log(x))

# plt.figure()
# plt.plot(np.arange(0.0009, 1, step=0.001), y)
# plt.grid()
# plt.show()


'''
Standars Distribution
'''
# x1 = np.random.normal(5, 0.1, size=(100,2))
# x2 = np.random.normal(10, 0.5, size=(100,2))

# x1_mean = np.sum(x1[:,0]) / len(x1)
# x2_mean = np.sum(x2[:,0]) / len(x2)

# x1_var = np.sum((x1[:,0] - x1_mean)**2) / len(x1)
# x2_var = np.sum((x2[:,0] - x2_mean)**2) / len(x2)

# x1_deviation = math.sqrt(x1_var)
# x2_deviation = math.sqrt(x2_var)

# x1_std = (x1 - x1_mean) / x1_deviation
# x2_std = (x2 - x2_mean) / x2_deviation

# plt.figure()
# plt.scatter(x1[:,0], x1[:,1])
# plt.scatter(x2[:,0], x2[:,1])
# plt.scatter(x1_std[:, 0], x1_std[:, 1])
# plt.scatter(x2_std[:, 0], x2_std[:, 1])
# plt.show()


'''
New way to index array
'''
# path = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
# idxs = np.random.randint(0, len(path), size=(10, ))
# print(idxs)

# path = path[idxs]
# print(path, path.shape)

'''
learning rate scheduler == learning rate annealing
'''
alpha = 0.01
epoch = 100
decay = alpha / epoch
e = 1

for i in np.arange(0, epoch):

    print("{} alpha = {}".format((i + 1), alpha))
    alpha = alpha * (1) / (1 + decay * e)