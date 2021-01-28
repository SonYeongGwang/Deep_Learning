import numpy as np
import matplotlib.pyplot as plt
import math

a = 6
b = 3
c = 2.0

print(a + b)
print(a + c)

A = [1, 2, 3]
B = [1, 2]

for a, b in zip(A, B):
    print(a, b)

y = []
## cross-entropy cost function
for x in np.arange(0.0009, 1, step=0.001):
    y.append(math.log(x))

plt.figure()
plt.plot(np.arange(0.0009, 1, step=0.001), y)
plt.grid()
plt.show()