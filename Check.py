from keras.models import load_model
import sys
from PIL import Image
from numpy import asarray
import pickle

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
#if len(sys.argv) > 1:
#    sampleImage=sys.argv[1]

model = load_model("Models\\" + "postim_weights.h5")

pickle_in = open("Models/Mappings.pickle","rb")
labels = pickle.load(pickle_in)
error = 0

for i in range (1, 2360):
    try:
        sampleImage = "images/"+str(i).zfill(4)+".jpg"
        image = Image.open(sampleImage)
        img_resized = image.resize((256,256))
        data = asarray(img_resized)/255.0 # rgb 0 - 255 to 0 - 1

        y = model.predict(data.reshape(1, 256, 256, 3))  # data reshape for a single image
        
        predicted_class_indices=np.argmax(y,axis=1)

        ind=labels[predicted_class_indices[0]]
        # print(predicted_class_indices, ind)
        if (str(ind).split()[0] != str(i)):  
            print("error at ", i, ind, error)
            error += 1
    except FileNotFoundError:
        print("file not found for " + str(i));     
    
print("error count: ", error)



