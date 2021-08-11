# The idea here is to convert a n dimentional chanel image to a single dimentional image
# for example if one RGB pixel has the value of (250,30,46)

import time
import cv2
import numpy as np
import sys
import numpy
from numpy import ndarray


def decimalToGCM(num : int) -> list : 
    out_num = []

    def get_quotients(num):
        if num >= 1:
            out_num.append(num % 256)
            get_quotients(num // 256)
    
    get_quotients(num)
    len_out_num = len(out_num)
    
    if len_out_num < 3:
        out_num = np.pad(out_num, (0,3-len_out_num), 'constant',constant_values=(0,0))
    
    return out_num[::-1]


def to_GCM(img : ndarray) -> ndarray : 
    return np.einsum('k,ijk->ij', np.array([256*256, 256, 1]), img)


def to_numpy(img : ndarray) -> ndarray : 

    h = img.shape[0]
    w = img.shape[1]
    ret_img = np.zeros((h, w, 3))

    for y in range(0, h):
        for x in range(0, w):
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
