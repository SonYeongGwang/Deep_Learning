from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
    (iH, iW) = image.shape[0:2]
    (kH, kW) = K.shape[0:2]

    pad = (kW - 1) // 2

    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    for j in np.arange(0, image.shape[0] - 2*pad):
        for i in np.arange(0, image.shape[1] - 2*pad):
            roi = image[j:j + 2*pad + 1:, i:i + 2*pad + 1]

            k = np.sum((K * roi))
            output[j, i] = k

    # for y in np.arange(pad, iH + pad):
    #     for x in np.arange(pad, iW + pad):
    #         roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

    #         k = (roi * K).sum()
    #         output[y - pad, x - pad] = k
    
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))

sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0], ), dtype="int")

laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int")

kernelBank = (
    ("small_blur", smallBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("emboss", emboss)
)

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernelName, K) in kernelBank:
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    cv2.imshow("Original", gray)
    cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()