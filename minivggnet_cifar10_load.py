from ImageTools import SimpleDatasetLoader
from ImageTools import SimplePreprocessor
from ImageTools import ImageArrayPreprocessor
from keras.datasets import cifar10
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2
from GpuConfig import GpuMemoryAllocate

GpuMemoryAllocate.SetMemoryGrowth()

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

print("[INFO] loading images...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
idxs = np.random.randint(0, len(testX), size=(5, ))
testX = testX[idxs]
testX = testX.astype("float") / 255.0

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = load_model(args["model"])

preds = model.predict(testX, batch_size=32).argmax(axis=1)

for (i, TestX) in enumerate(testX):
    print("Label: {}".format(labelNames[preds[i]]))
    cv2.imshow("Image & predictions", TestX)
    cv2.waitKey(0)