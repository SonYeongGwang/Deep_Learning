from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from ImageTools import SimplePreprocessor
from ImageTools import SimpleDatasetLoader
from ImageTools import ImageArrayPreprocessor
from sklearn.metrics import classification_report
from ImageTools.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

epoch = 50

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
data = data.astype("float") / 255.0

labelNames = ["cat", "dog"]

labelsOneHot = []

for label in labels:
    labelOneHot = [1, 0] if label == 'cats' else [0, 1]
    labelsOneHot.append(labelOneHot)

labelsOneHot = np.array(labelsOneHot)

print("[INFO] compling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(data, labelsOneHot, epochs=epoch, verbose=1, batch_size=32)
# 32 images will be presented to the network

print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] evaluating network...")
predictions = model.predict(data, batch_size=32)
print(predictions)
print(classification_report(labelsOneHot.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

plt.style.use("dark_background")
plt.figure()
plt.plot(np.arange(0, epoch), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, epoch), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("The number of Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()