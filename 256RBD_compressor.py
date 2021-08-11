# The idea here is to convert a n dimentional chanel image to a single dimentional image
# for example if one RGB pixel has the value of (250,30,46)

import time
import cv2
import numpy as np
import sys
import numpy
from numpy import ndarray


def decimalToGCM(num):
    out_num = []
    def get_quotients(num):
        if num >= 1:
            out_num.append(num % 256)
            get_quotients(num // 256)
    get_quotients(num)
    if len(out_num)==2:
        out_num.append(0)
    elif len(out_num)==1:
        out_num.append(0)
        out_num.append(0)
    elif len(out_num)==0:
        out_num.append(0)
        out_num.append(0)
        out_num.append(0)

    return out_num[::-1]


def to_GCM(img):
    return np.einsum('k,ijk->ij', np.array([256*256, 256, 1]), img)


def to_numpy(img):
    ret_img = np.zeros((img.shape[0], img.shape[1], 3))
    # grab the image dimensions
    h = img.shape[0]
    w = img.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            # print(decimalToGCM(img[y, x]))
            ret_img[y, x] = np.array(decimalToGCM(img[y, x]))

    # return the thresholded image
    return ret_img

# Read an image from filesystem
img = cv2.imread(r"images\image1.jpg")

# Convert to 256RDB form
new_img = np.einsum('k,ijk->ij', np.array([256*256, 256, 1]), img)

# Convert back to numpy array
reversedimg = to_numpy(new_img)

# Finding errors in compresion and extraction
(unique, counts) = numpy.unique(reversedimg-img, return_counts=True)
frequencies = numpy.asarray((unique, counts)).T
print(frequencies)