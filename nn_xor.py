from ImageTools.pyimagesearch import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
losses = []
epoch = 20000

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
losses = nn.fit(X, y, epochs=epoch)

for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    # print(np.shape(nn.predict(x)))
    # print(np.shape(nn.predict(x)[0]))
    # print(nn.predict(x)[0][0])
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch), losses)
plt.title("Training Loss")
plt.xlabel("# of Epoch")
plt.ylabel("Loss")
plt.show()