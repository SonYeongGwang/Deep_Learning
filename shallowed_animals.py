from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ImageTools import SimplePreprocessor
from ImageTools import ImageArrayPreprocessor
from ImageTools import SimpleDatasetLoader
from ImageTools.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

epoch = 100

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageArrayPreprocessor()

sd = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sd.load(imagePaths, Verbose=200)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=None)

trainY_OneHots = []
testY_OneHots = []

for Label in trainY:
    OneHot = [1, 0] if Label == 'cats' else [0, 1]
    trainY_OneHots.append(OneHot)

for Label in testY:
    OneHot = [1, 0] if Label == 'cats' else [0, 1]
    testY_OneHots.append(OneHot)

trainY_OneHots = np.array(trainY_OneHots)
testY_OneHots = np.array(testY_OneHots)
# print(trainY.shape, trainY)
# lb = LabelBinarizer()
# trainY = lb.fit_transform(trainY)
# print(trainY.shape, trainY, "\n")
# testY = lb.fit_transform(testY)

print("[INFO] compling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY_OneHots, validation_data=(testX, testY_OneHots), epochs=epoch, verbose=1, batch_size=32)
# 32 images will be presented to the network

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
# print(predictions)
# print(classification_report(testY_OneHots.argmax(axis=1), predictions.argmax(axis=1), labels=["cat", "dog"]))

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
plt.savefig("output/shallowed_animals.png")