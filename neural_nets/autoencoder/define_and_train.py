from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import imageio
import os

# Definfing the dataset path
cwd = Path(r'C:\Users\Atharva\Projects\Image-Compression-Techniques\Dataset')

dataset = []

# Creating the dataset
for index in range(1, 2300):
    img_index = str(index)
    filename = cwd / 'data' / f'kodim{img_index}.png'
    try:
        img_data = imageio.imread(filename)
    except Exception:
        continue

    # Normalizing and appending to the dataset
    dataset.append(img_data/255)

dataset = np.array(dataset)
print(dataset.shape)
