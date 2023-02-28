# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 23:50:48 2023

@author: rajsu
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model=Sequential()
model.add(Convolution2D(64,3,3,input_shape=(175,175,3),activation=("relu")))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64,3,3,activation=("relu")))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation=("relu")))
model.add(Dense(128,activation=("relu")))
model.add(Dense(128,activation=("relu")))
model.add(Dense(1,activation='Softmax'))
model.compile(optimizer="adam",loss="BinaryCrossentropy",metrics=(["Accuracy"]))
"""here we use the loss = binarycrossentropy beacuse we classfier into two categeries
if we need to classfier more than two categories then we use CategoricalCrossentropy"""

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('D:/practiceml/CNN/dataset/training_set',
                                                target_size=(175, 175),
                                                batch_size=32,
                                                class_mode='binary')

testing_set = test_datagen.flow_from_directory('D:/practiceml/CNN/dataset/test_set',
                                                target_size=(175, 175),
                                                batch_size=32,
                                                class_mode='binary')

model.fit(
        training_set,
        epochs=25,
        validation_data=testing_set)

dir_list = 'D:/practiceml/CNN/dataset/check_set'
import os
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt
import numpy as np

for i in os.listdir(dir_list):
    img=image.load_img(dir_list+'//'+i,target_size=(175,175))
    plt.imshow(img)
    plt.show()
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val=model.predict(images)
    if(val==0):
        print("this is cats")
    elif(val==1):
        print("this is dog")




