import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np

image = load_img('/home/a/animals/test_set/test_set/cats/cat.4001.jpg')
image = img_to_array(image)
orin = cv2.imread('/home/a/animals/test_set/test_set/cats/cat.4001.jpg')
print(np.expand_dims(image, axis=0).shape)
print(orin, np.shape(orin))

cv2.imshow('test', image)
cv2.imshow('orin', orin)
cv2.waitKey(0)