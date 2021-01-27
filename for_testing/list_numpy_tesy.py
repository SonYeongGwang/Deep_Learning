import numpy as np

np.random.seed(1)

'''
######append test######
w = []
a = np.random.randn(4, 2)
a2 = np.random.randn(2, 3)
a3 = np.random.randn(4, 2)

w.append(a)
print(w, np.shape(w), "\n")
w.append(a2)
print(w, np.shape(w), "\n")
w.append(a3)
print(w, np.shape(w), "\n")
print(w[1])
'''

# a = np.random.randn(4)
# print(a, np.shape(a))   #shape: (4, )

# a_1 = np.random.randn(3)

# a_2 = [np.atleast_2d(a)]
# print("a_2: ", a_2, np.shape(a_2))
# print(a_2[0])

# b = np.random.randn(4, 1)
# print(b, np.shape(b), len(b))   #shape: (4, 1)  length: 4

# c = np.random.randn(4, 1)
# print(c)

# d = c + a       # --> shape: (4, 4)
#                 # [[[c[0] + a[0], c[0] + a[1], c[0] + a[2], c[0] + a[3]]
#                 #   [c[1] + a[0], c[1] + a[1], c[1] + a[2], c[1] + a[3]]
#                 #   [c[2] + a[0], c[2] + a[1], c[2] + a[2], c[2] + a[3]]
#                 #   [c[3] + a[0], c[3] + a[1], c[3] + a[2], c[3] + a[3]]]
# e = c + b       # --> shape: (4, 1)

# print(np.shape(d), np.shape(e))
# print(d)

# f = a + a
# print(np.shape(f), f)      #shape: (4, )

# print(np.shape(a_1 + b))    #shape: (4, 3)

# x = np.random.randn(4, 2)
# y = np.random.randn(4, 1)

# for (X, Y) in zip(x, y):
#     print(X, Y)
#     print("1")
    
# print(X, Y)

# D = np.array([[[1, 2, 3], [4, 5, 6]],
#               [[7, 8, 9], [10, 11, 12]]])
# print(np.shape(D))
# D = D[: : -1]
# print(D)
# The general syntax for a slice is array[start:stop:step]. Any or all of the values start, stop, and step may be left out (and if step is left out the colon in front of it may also be left out):