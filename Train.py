# programming marko.rantala@gmail.com
# 29.04.2019

from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import pickle
import sys

import matplotlib.pyplot as plt

epochs = 100
initial_epochs = 16
if len(sys.argv) > 1:
    epochs = int(sys.argv[1])
if len(sys.argv) > 2:
    initial_epochs = int(sys.argv[2])
    
traindf=pd.read_csv("./postimerkit2014.csv",dtype=str,sep=';' )

print(traindf.head())
print(traindf.shape)
#testdf=pd.read_csv("./sampleSubmission.csv",dtype=str)
#traindf["id"]=traindf["id"].apply(append_ext)
#testdf["id"]=testdf["id"].apply(append_ext)
#datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)
datagen=ImageDataGenerator(rescale=1./255.)  # how about data augmentation
# datagen=ImageDataGenerator(rescale=1./255.,
    # rotation_range=3,   # think those, augmented features .. 
    # width_shift_range=0.03,
    # height_shift_range=0.03
# )  # how about data augmentation)  # let's do that later here

train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./images/",
    x_col="id",
    y_col="label",
    #subset="training", # all
    #drop_duplicates=False, #unknown parameter!???
    seed=42,
    batch_size=40,  # so full set  is dividable with this 2320/40 = 58
    shuffle=True,
    class_mode="categorical",
    target_size=(256,256))  # small resize smaller from 260*260


print(train_generator)

# visualize augmented points, actually not augmented yet, lets do that later, maybe add noise, turn and so on.. 

def show_image_data_sample(train_generator):
    plt.figure(figsize=(6, 6))
    (X_batch, Y_batch) = train_generator.next()
    print(X_batch.shape, Y_batch.shape )
    print(Y_batch)  # labels

    for i in range(9):
        plt.subplot(3, 3, (i + 1))
        #plt.imshow(X_batch[i, :, :, 0], cmap=plt.get_cmap('gray'), interpolation='none')
        plt.imshow(X_batch[i])

    train_generator.reset()  # as used once, can have strange orders otherwise
    plt.show()
    
show_image_data_sample(train_generator)
(X_batch, Y_batch) = train_generator.next()   # just to get size of training set
classes= Y_batch.shape[1]
 
#save labels for later usage, mappings back to info
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
pickle_out = open("Models/Mappings.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()


# build the model, not so deep needed !

initializer='glorot_uniform'  # probably the best and default
activation='relu' # try other activations too like 'tanh'

model = Sequential()
model.add(Conv2D(12, (2, 2), padding='same', kernel_initializer=initializer, activation='relu',
                 input_shape=(256,256,3)))
#model.add(Activation('relu'))

model.add(Conv2D(16, (3, 3), kernel_initializer=initializer, activation='relu'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(AveragePooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

#model.add(Conv2D(64, (3, 3), padding='same'))
#model.add(Activation('relu'))
model.add(Conv2D(20, (3, 3), kernel_initializer=initializer))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(401, activation='relu'))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(classes, activation='softmax', kernel_initializer=initializer))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"]) 

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

# real training here, not modified images so those start to converge fast 
history_callback = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    #validation_data=valid_generator,
                    #validation_steps=STEP_SIZE_VALID,
                    epochs=initial_epochs  # initial training 
)


model.save("./Models/postim_weights.h5")    

def draw_stats():
        
        
    plt.plot( history_callback.history["loss"], color='blue', label='loss')
    plt.plot( history_callback.history["accuracy"], color='red', label='accuracy')
    #plt.plot( best_val_loss_history.history["mean_absolute_error"], color='red', label='accuracy (mae)')
    #plt.plot( best_val_loss_history.history["val_loss"], color='magenta', label='val_loss')
    #plt.plot( best_val_loss_history.history["val_mean_absolute_error"], color='green', label='val_acc (mae)')
    
    plt.legend(loc='best')
    #print (x, y)
    plt.show()
    
draw_stats()


# second set with modified images
datagen=ImageDataGenerator(rescale=1./255.,
    rotation_range=3,   # think those, as augmented features .. 
    width_shift_range=0.03,
    height_shift_range=0.03
)  # how about data augmentation)  # how about data augmentation

train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./images/",
    x_col="id",
    y_col="label",
    #subset="training", # all
    #drop_duplicates=False, #unknown parameter!???
    seed=42,
    batch_size=40,  # so full set  is dividable with this 2320/40 = 58
    shuffle=True,
    class_mode="categorical",
    target_size=(256,256))  # small resize smaller from 260*260

show_image_data_sample(train_generator)


history_callback = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    #validation_data=valid_generator,
                    #validation_steps=STEP_SIZE_VALID,
                    epochs=epochs
)


model.save("./Models/postim_weights.h5")    

draw_stats()
