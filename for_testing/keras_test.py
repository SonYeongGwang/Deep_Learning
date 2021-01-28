from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# image = cv2.imread('/home/a/animals/training_set/training_set/cats/cat.2.jpg')

# height, width, depth = image.shape[0], image.shape[1], image.shape[-1]

# inputShape = (height, width, depth)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
# model.add(Activation('relu'))
# print(model.output_shape)
# model.add(Flatten())
# print(model.output_shape)

labels = np.array(['yes', 'no', 'no', 'yes'])
labelsOneHot = []
lb = LabelBinarizer()
lb_output = lb.fit_transform(labels)
print(lb_output, lb.classes_)

for label in labels:
    labelOneHot = [0, 1] if label == 'yes' else [1, 0]
    labelsOneHot.append(labelOneHot)

labelsOneHot = np.array(labelsOneHot)
print(labelsOneHot)

print(labelsOneHot.argmax(axis=1))

# lb_output2 = lb.fit_transform(['c', 'd', 'a', '1'])
# print(lb_output2, lb.classes_)
# 1. numerical -> 2. alphbetical order