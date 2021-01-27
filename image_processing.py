import cv2
from ImageTools import SimplePreprocessor
from keras.preprocessing.image import img_to_array

image = cv2.imread('/home/a/animals/training_set/training_set/cats/cat.2.jpg')
image_array = img_to_array(image)
cv2.imshow('image', image)
cv2.waitKey()

print(image_array.shape)
print(image.shape)

p = SimplePreprocessor(32,32)
image = p.preprocess(image)
cv2.imshow('image', image)
cv2.waitKey()