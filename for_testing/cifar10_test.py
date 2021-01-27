# from keras.datasets import cifar10
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# ((trainX, trainY), (testX, testY)) = cifar10.load_data()
# print(np.shape(trainX), np.shape(trainY))

target = [[1, 3, 5, 2, 1]]
print(np.shape(np.transpose(target)))
target = np.transpose(target)
print(target)
lb = LabelBinarizer()

target_onehot = lb.fit_transform(target)
print(target_onehot)