# The idea here is to convert a n dimentional chanel image to a single dimentional image
# for example if one RGB pixel has the value of (250,30,46) will
# get converted to 250*256*256 + 30*256 + 46  = 16391726

from numpy import ndarray
import numpy as np
import pickle
import numpy
import time
import sys
import cv2


def decimalToGCM(num: int) -> list:
    """Convert decimal numbers to GCM representation

    Args:
        num (int): input number

    Returns:
        list: GCM representation 
    """
    out_num = []

    def get_quotients(num : int) -> None:
        """Returns quotients 

        Args:
            num (int): input number to be converted to GCM representation
        """
        if num >= 1:
            out_num.append(num % 256)
            get_quotients(num // 256)

    get_quotients(num)
    len_out_num = len(out_num)

    if len_out_num < 3:
        out_num = np.pad(out_num, (0, 3-len_out_num),
                         'constant', constant_values=(0, 0))

    return out_num[::-1]


def to_GCM(img: ndarray) -> ndarray:
    """Converts a GCM image array to decimal representation.

    Args:
        img (ndarray): input image

    Returns:
        ndarray: GCM representation
    """
    return np.einsum('k,ijk->ij', np.array([256*256, 256, 1]), img)


def to_numpy(img: ndarray) -> ndarray:
    """Converts GCM representation to decimal representation

    Args:
        img (ndarray): GCM representation

    Returns:
        ndarray: image array
    """

    h = img.shape[0]
    w = img.shape[1]
    ret_img = np.zeros((h, w, 3))

    for y in range(0, h):
        for x in range(0, w):
            ret_img[y, x] = np.array(decimalToGCM(img[y, x]))

    # return the thresholded image
    return ret_img

def save_to_file(gcm_list : ndarray, filepath : str) -> None:
    """Saves decimal representation to file

    Args:
        gcm_list (ndarray): input GCM numpy list
        filepath (str): file path to save in 
    """
    with open(filepath, "wb") as f:
        pickle.dump(gcm_list, f)

    print("Saved to :",filepath)

def read_from_file(filepath : str) -> ndarray:
    """Retruns decimal representation from saved file

    Args:
        filepath (str): [description]

    Returns:
        ndarray: [description]
    """

    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Read an image from filesystem
img = cv2.imread(r"images\image1.jpg")

# Convert to 256RDB form
new_img = np.einsum('k,ijk->ij', np.array([256*256, 256, 1]), img)

# Save to file:
save_to_file(new_img,r'C:\Users\Atharva\Desktop\Projects\Image-Compression-Techniques\images\out.pkl')

# Convert back to numpy array
reversedimg = to_numpy(new_img)

# Finding errors in compresion and extraction
(unique, counts) = numpy.unique(reversedimg-img, return_counts=True)
frequencies = numpy.asarray((unique, counts)).T
print(frequencies)


