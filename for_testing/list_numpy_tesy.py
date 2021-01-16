import numpy as np

np.random.seed(1)
a = np.random.randn(4)
print(a, np.shape(a))   #shape: (4, )

b = np.random.randn(4, 1)
print(b, np.shape(b))   #shape: (4, 1)

c = np.random.randn(4, 1)
print(c)

d = c + a       # --> shape: (4, 4)
                # [[[c[0] + a[0], c[0] + a[1], c[0] + a[2], c[0] + a[3]]
                #   [c[1] + a[0], c[1] + a[1], c[1] + a[2], c[1] + a[3]]
                #   [c[2] + a[0], c[2] + a[1], c[2] + a[2], c[2] + a[3]]
                #   [c[3] + a[0], c[3] + a[1], c[3] + a[2], c[3] + a[3]]]
e = c + b       # --> shape: (4, 1)

print(np.shape(d), np.shape(e))
print(d)

f = a + a
print(np.shape(f))      #shape: (4, )