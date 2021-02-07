# import matplotlib
# matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from ImageTools.nn.conv import MiniVGGNet
from keras.datasets import cifar10
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
from GpuConfig import GpuMemoryAllocate

GpuMemoryAllocate.SetMemoryGrowth()

epoch = 40

# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True, help="path to output model")
# ap.add_argument("-o", "--output", required=True, help = "path to the output loss/acc plot")
# args = vars(ap.parse_args())

print("[INFO] loading images...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print("[INFO] compling model...")
opt = SGD(lr=0.01, momentum=0.9, decay= (0.01 / epoch), nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epoch, verbose=1, batch_size=64)
# 32 images will be presented to the network

# print("[INFO] serializing network...")
# model.save(args["model"])

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(predictions)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

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
plt.show()
# plt.savefig(args["output"])