from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from ImageTools import SimplePreprocessor
from ImageTools import SimpleDatasetLoader
from ImageTools import ImageArrayPreprocessor
from ImageTools.nn.conv import ShallowNet
from keras.datasets import cifar10
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

epoch = 50
opt = SGD(0.01)

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = ShallowNet.build(width=32, height=32, depth=3, classes=len(labelNames))
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
H = model.fit(trainX, trainY, batch_size=32, epochs=epoch, verbose=1, validation_data=(testX, testY))

predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),target_names=labelNames))

plt.style.use("dark_background")
plt.figure()
plt.plot(np.arange(0, epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0 ,epoch), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epoch), H.history["val_acc"], label="val_acc")
plt.xlabel("The number of Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("output/shallowed_cifar10.png")