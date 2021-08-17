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

# Defifing the model 

# Input layer
input_img = Input(shape=dataset[0].shape)


x = Conv2D(16, (7, 7), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(10, (3, 3), activation='relu', padding='same')(x)

# Encoder Model  
encoded = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2D(10, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (7, 7), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

# Dencoder Model  
decoded = Conv2D(3, (7, 7), activation='sigmoid', padding='same')(x)

# Autoencoder Model
autoencoder = Model(input_img, decoded)

# Compiling the model 
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(autoencoder.summary())


# Training 
autoencoder.fit(dataset, dataset,
                epochs=1000,
                batch_size=2,
                shuffle=True)

# Evaluation
predicted_imgs = autoencoder.predict(dataset)
for i in range(predicted_imgs.shape[0]):
    plt.figure()
    img_uint = (255*predicted_imgs[i]).astype(np.uint8)
    plt.imshow(predicted_imgs[i])
    imageio.imsave(cwd / 'results' / f'result-{i+1}.png', img_uint)
plt.show()