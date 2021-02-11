import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from ImageTools import SimplePreprocessor
from ImageTools import SimpleDatasetLoader
from ImageTools import ImageArrayPreprocessor
from sklearn.metrics import classification_report
from ImageTools.nn.conv import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
from GpuConfig import GpuMemoryAllocate
GpuMemoryAllocate.SetMemoryGrowth()

def step_dacay(epoch):
    initAlpha = 0.01
    factor = 0.5
    dropEvery = 10

    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    return alpha

epoch = 100

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

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

print("[INFO] compling model...")

callback = [LearningRateScheduler(step_dacay)]

opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, labelsOneHot_train, validation_data=(testX, labelsOneHot_test),epochs=epoch
                , verbose=1, batch_size=64, callbacks=callback)
# 64 images will be presented to the network

print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(labelsOneHot_test.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

plt.style.use("dark_background")
plt.figure()
plt.plot(np.arange(0, epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epoch), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("The number of Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('/home/a/Deep_Learning/output/minivggnet_animals_lr2.png')