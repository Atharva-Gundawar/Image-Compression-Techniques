# The idea here is to convert a n dimentional chanel image to a single dimentional image
# for example if one RGB pixel has the value of (250,30,46)
import time
import cv2
import numpy as np
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

img = cv2.imread(r"images\image1.jpg")

def to_GCM(img):
    pass

def to_numpy(img):
    pass

new_img = np.einsum('k,ijk->ij', np.array([256*256, 256, 1]), img)

len_img = len(str(img))
len_new = len(str(new_img))
print("Size reduced by :",'{:.1%}'.format(abs(len_img - len_new)/len_img))


# del len_img
# del len_new
# del img