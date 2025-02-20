from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers.core import Dense
from ImageTools import SimplePreprocessor
from ImageTools import SimpleDatasetLoader
from ImageTools import ImageArrayPreprocessor
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.optimizers import SGD
from keras.layers.core import Dropout
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from keras import backend as K

# image = cv2.imread('/home/a/animals/training_set/training_set/cats/cat.2.jpg')

# height, width, depth = image.shape[0], image.shape[1], image.shape[-1]

# inputShape = (height, width, depth)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
# model.add(Activation('relu'))
# print(model.output_shape)
# model.add(Flatten())
# print(model.output_shape)

# labels = np.array(['yes', 'no', 'no', 'yes'])
# labelsOneHot = []
# lb = LabelBinarizer()
# lb_output = lb.fit_transform(labels)
# print(lb_output, lb.classes_)

# for label in labels:
#     labelOneHot = [0, 1] if label == 'yes' else [1, 0]
#     labelsOneHot.append(labelOneHot)

# labelsOneHot = np.array(labelsOneHot)
# print(labelsOneHot)

# print(labelsOneHot.argmax(axis=1))


# imagePaths = list(paths.list_images("/home/a/animals/test_set/test_set/"))
# print(imagePaths)
# sp = SimplePreprocessor(32, 32)
# iap = ImageArrayPreprocessor()

# sd1 = SimpleDatasetLoader(preprocessors=[sp, iap])
# (data, labels) = sd1.load(imagePaths, Verbose=500)
# print(labels)

# labelsOneHot = []
# lb = LabelBinarizer()

# for label in labels:
#     labelOneHot = [1, 0] if label == 'cats' else [0, 1]
#     labelsOneHot.append(labelOneHot)

# labelsOneHot = np.array(labelsOneHot)
# print(labelsOneHot)


# print(labelsOneHot.argmax(axis=1))



# lb_output2 = lb.fit_transform(['c', 'd', 'a', '1'])
# print(lb_output2, lb.classes_)
# 1. numerical -> 2. alphbetical order

'''checking layer output (basic)'''
# training_data = np.array([[1], [2], [3], [4]])
# result_data = np.array([[2], [4], [6], [8]])

# model = Sequential()
# model.add(Dense(1, activation="linear", input_shape=(1,)))
# model.compile(loss="mean_squared_error", optimizer="SGD")
# model.fit(training_data, result_data, epochs=20, verbose=0)

# outputs = []
# for layer in model.layers:
#     keras_function = K.function([model.input], [layer.output])
#     outputs.append(keras_function([training_data, 1]))
# print(outputs[0][0][0][0])
# print(outputs[0][0][0][0]) --> this one may useful in the future
'''checking layer output (basic) END'''


'''checking layer output (advanced)'''
imagePaths = list(paths.list_images("/home/a/animals/test_set/test_set/"))

sp = SimplePreprocessor(32, 32)
iap = ImageArrayPreprocessor()

sd1 = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sd1.load(imagePaths, Verbose=500)
data = data.astype("float") / 255.0

labelsOneHot = []

for label in labels:
    labelOneHot = [1, 0] if label == 'cats' else [0, 1]
    labelsOneHot.append(labelOneHot)

labelsOneHot = np.array(labelsOneHot)

opt = SGD(lr=0.005)
model = Sequential()
inputShape = (32, 32, 3)
model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(data, labelsOneHot, epochs=10, verbose=1, batch_size=32)


outputs = []
keras_function = K.function([model.input], [model.layers[-1].output])
outputs.append(keras_function([data, 1]))
print(outputs)
# print(outputs[0][0][0][0]) --> this one may useful in the future
'''checking layer output (advanced) END'''