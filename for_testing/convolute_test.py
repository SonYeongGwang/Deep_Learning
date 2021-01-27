import numpy as np
from skimage.exposure import rescale_intensity

pad = 4

k = np.array([0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5])
k = np.atleast_2d(k)

sample_image = np.array([8, 8, 8, 8, 1, 2, 3, 4, 5, 6, 9, 9, 9, 9])
sample_image = np.atleast_2d(sample_image)

ih, iw = sample_image.shape[0:2]
#  == sample_image.shape[1], sample_image.shape[0]
kh, kw = k.shape[0:2]

outputs = np.atleast_2d(np.zeros((sample_image.shape[0], sample_image.shape[1]-2*pad)))

print(outputs)


for j in np.arange(0, iw - 2*pad):
    roi = sample_image[:, j:j + 2*pad + 1]
    K = (k * roi).sum()
    outputs[:,j] = K
    print(np.shape(outputs), outputs.astype("uint8"))
