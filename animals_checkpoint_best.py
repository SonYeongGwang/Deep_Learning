from sklearn.model_selection import train_test_split
from ImageTools import SimplePreprocessor
from ImageTools import SimpleDatasetLoader
from ImageTools import ImageArrayPreprocessor
from ImageTools.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import os
import numpy as np
import argparse
from GpuConfig import GpuMemoryAllocate
from keras.callbacks import ModelCheckpoint

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help = "path to the dataset")
ap.add_argument("-w", "--weights", required=True, help="path to best model weights file")
args = vars(ap.parse_args())

epoch = 50

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageArrayPreprocessor()

sd1 = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sd1.load(imagePaths, Verbose=500)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

labelNames = ["cat", "dog"]

labelsOneHot_train = []
labelsOneHot_test = []

for label in trainY:
    labelOneHot = [1, 0] if label == 'cats' else [0, 1]
    labelsOneHot_train.append(labelOneHot)

for label in testY:
    labelOneHot = [1, 0] if label == 'cats' else [0, 1]
    labelsOneHot_test.append(labelOneHot)

labelsOneHot_train = np.array(labelsOneHot_train)
labelsOneHot_test = np.array(labelsOneHot_test)

print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(32, 32, 3, 2)
model.compile(optimizer=opt, loss = "categorical_crossentropy", metrics=["accuracy"])

fname = os.path.sep.join([args["weights"], "animals_best_weight.h5"])

checkpoint = ModelCheckpoint(fname, save_best_only=True, verbose=1)
callback = [checkpoint]

print("[INFO] training network...")
model.fit(trainX, labelsOneHot_train, batch_size=64, epochs=epoch, callbacks=callback, validation_data=(testX, labelsOneHot_test))