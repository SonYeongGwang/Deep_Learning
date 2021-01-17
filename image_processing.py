import cv2
from ImageTools import SimplePreprocessor

image = cv2.imread('/home/a/animals/training_set/training_set/cats/cat.2.jpg')
cv2.imshow('image', image)
cv2.waitKey()

print(image.shape)

p = SimplePreprocessor(32,32)
image = p.preprocess(image)
cv2.imshow('image', image)
cv2.waitKey()