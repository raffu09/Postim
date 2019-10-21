from keras.models import load_model
import sys
from PIL import Image
from numpy import asarray
import pickle
import os

from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


sampleImage = "samples/sampleImage.jpg" # 
if len(sys.argv) > 1:
    sampleImage=sys.argv[1]

image = Image.open(sampleImage)
img_resized = image.resize((256,256))
data = asarray(img_resized)/255.0 # rgb 0 - 255 to 0 - 1

modelFile = os.path.join("Models", "postim_weights.h5")
model = load_model(modelFile)

y = model.predict(data.reshape(1, 256, 256, 3))  # data reshape for a single image

print(y)

predicted_class_indices=np.argmax(y,axis=1)

print(predicted_class_indices)

pickle_in = open("Models/Mappings.pickle","rb")
labels = pickle.load(pickle_in)

print(labels[predicted_class_indices[0]])




