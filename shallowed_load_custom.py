from ImageTools import SimpleDatasetLoader
from ImageTools import SimplePreprocessor
from ImageTools import ImageArrayPreprocessor
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

labelNames = ["cat", "dog"]

print("[INFO] sampling images...")
imagePath = ['/home/a/Deep_Learning/image3.jpg']

sp = SimplePreprocessor(32, 32)
iap = ImageArrayPreprocessor()
imagePath
sd1 = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sd1.load(imagePath)
data = data.astype("float") / 255.0

print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

print("[INFO] prediction...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

image = cv2.imread(imagePath[0])
cv2.putText(image, "Label: {}".format(labelNames[preds[0]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 220, 10))
cv2.imshow("Image & predictions", image)
cv2.waitKey(0)