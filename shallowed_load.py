from ImageTools import SimpleDatasetLoader
from ImageTools import SimplePreprocessor
from ImageTools import ImageArrayPreprocessor
from keras.datasets import cifar10
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("[INFO] sampling images...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
idxs = np.random.randint(0, len(testX), size=(5, ))
testX = testX[idxs]

testX = testX.astype("float") / 255.0

print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

print("[INFO] prediction...")
preds = model.predict(testX, batch_size=32).argmax(axis=1)

for (i, TestX) in enumerate(testX):
    # cv2.putText(TestX, "Label: {}".format(labelNames[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 200, 10))
    print("Label: {}".format(labelNames[preds[i]]))
    cv2.imshow("Image & predictions", TestX)
    cv2.waitKey(0)