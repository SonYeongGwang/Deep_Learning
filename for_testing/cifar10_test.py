from keras.datasets import cifar10
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import cv2

((trainX, trainY), (testX, testY)) = cifar10.load_data()
print(np.shape(trainX), np.shape(trainY))
idxs = np.random.randint(0, len(trainX), size=(3, ))
print(testY)
trainX = trainX[idxs]

# for (i, TrainX) in enumerate(trainX):
#     print(np.shape(TrainX))
#     cv2.imshow("trainX", TrainX)
#     cv2.waitKey(0)

# target = [[1, 3, 5, 2, 1]]
# print(np.shape(np.transpose(target)))
# target = np.transpose(target)
# print(target)
lb = LabelBinarizer()

target_onehot = lb.fit_transform(testY)
print(target_onehot)