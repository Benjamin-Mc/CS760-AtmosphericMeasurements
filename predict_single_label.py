import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, UnidentifiedImageError
from pathlib import Path

import tensorflow as tf
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from IPython.display import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Specify csv file with humidity temperature and picture information
df = pd.read_csv("output_10000.csv")
# The 5 columns here can use humidity and temperature. Consider extracting the time in imageDate separately and also use it as a feature.
humidity_feature = df['humidity'].values
temperature_feature = df['temperature C 2m'].values

# unzip
with zipfile.ZipFile("/content/drive/My Drive/760/save_10000.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/drive/My,Drive/760/")

# choose size 100
humidity_feature = humidity_feature[:100]
temperature_feature = temperature_feature[:100]
print(df.iloc[1][2])

#finds nth occurence of character in a string
def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

gray_images = [] # Store grayscale image -- currently not used
image_feature = []

#finds nth occurence of character in a string
def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

#200 images in this set total - change range to see more
for i in range(100):
    imgPath = "F:\760\CNN\CS760-AtmosphericMeasurements\save_10000/" + df.iloc[i][2]
    #images with folders that are single digits are saved as 05 in the path, but the folder name is just 5
    #so we look for these images paths and cut out the leading 0
    # Here we get the index of the sixth / and then check whether there is a 0 behind it. If there is, delete the 0 to ensure that the folder can be opened.
    i = find_nth(imgPath, "/", 5)
    if imgPath[i+1] == "0":
        imgPath = imgPath[:i+1] + imgPath[i+2:]
    try:
        #images are different sizes so we will likely need to standardize the size
        img = Image.open(Path(imgPath))
        # Check if the image is color (has 3 channels) or grayscale (only has 1 channel)
        if np.array(img).shape[-1] == 3:
            # Color images, resized and stored in color_images
            imgResize = transforms.Resize(size=(400,400))(img)
            image_feature.append(np.array(imgResize))
        else:
            # Grayscale image, resized and stored in gray_images
            imgResize = transforms.Resize(size=(400,400))(img)
            gray_images.append(np.array(imgResize))

            imgResize = transforms.Resize(size=(400, 400))(img.convert('RGB'))  # Convert grayscale image to RGB
            image_feature.append(np.array(imgResize))

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(imgResize)
        # plt.show()

    except (UnidentifiedImageError, FileNotFoundError) as e:
        # An UnidentifiedImageError exception occurs, the wrong path is logged and the loop continues
        print(f"Error processing image: {imgPath}")
        continue

image_feature = np.array(image_feature)

# Standardize humidity features using StandardScaler
# scaler = StandardScaler()
# humidity_feature = scaler.fit_transform(humidity_feature.reshape(-1, 1))

# split data
X_train_img, X_test_img, X_train_hum, X_test_hum = train_test_split(image_feature, humidity_feature, test_size=0.2, random_state=42)

# The input layer receives image_feature
image_input = Input(shape=(400, 400, 3))
conv1 = Conv2D(32, (3, 3), activation='relu')(image_input)
conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
conv3 = Conv2D(128, (3, 3), activation='relu')(conv2)
# conv4 = Conv2D(256, (3, 3), activation='relu')(conv3)
maxpool1 = MaxPooling2D((2, 2))(conv3)
# Add more convolution and pooling layers and other layers? ? ?

flatten = Flatten()(maxpool1)
image_dense = Dense(256, activation='relu')(flatten)

# Connect to output layer
output = Dense(1, activation='linear')(image_dense)  # Use a linear activation function since humidity is a continuous value

model = Model(inputs=image_input, outputs=output)

# complie model
model.compile(optimizer='adam', loss='mean_squared_error')  # Use mean square error as loss function

# Prepare training data, including images and corresponding humidity labels
# image_feature is image data
# humidity_labels is the corresponding humidity label

# fit model
model.fit(X_train_img, X_train_hum, epochs=10, batch_size=32)

# Draw model structure diagram
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

# show on colab

Image(filename='model_structure.png')

# predict
predictions = model.predict(X_test_img)


# MSE
mse = mean_squared_error(X_test_hum, predictions)

# RMSE
rmse = np.sqrt(mse)

# MAE
mae = mean_absolute_error(X_test_hum, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
