import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha
    
    def pr(self):
        print(self.alpha)