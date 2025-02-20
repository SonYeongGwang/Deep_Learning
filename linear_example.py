import numpy as np
import cv2

labels = ['dogs', 'cats']
np.random.seed(1)

W = np.random.randn(len(labels), 3072)
# b = np.random.randn(len(labels), 1)    --> it will make scores to have a shape of (2, 2)
b = np.random.randn(len(labels))

orig = cv2.imread("./image3.jpg")
image = cv2.resize(orig, (32, 32)).flatten()

scores = W.dot(image) + b

for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Image", orig)
cv2.waitKey()