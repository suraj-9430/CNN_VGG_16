# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 21:07:25 2023

@author: Suraj Raj
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

loaded_model=pickle.load(open('D:/practiceml/CNN/cat_dog.sav','rb'))

my_image = load_img('D:/practiceml/CNN/dataset/check_set/dog.4013.jpg', target_size=(224, 224))
plt.imshow(my_image)
plt.show()
#preprocess the image
my_image = img_to_array(my_image)
my_image = my_image/255
my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
#my_image = preprocess_input(my_image)

#make the prediction
prediction = loaded_model.predict(my_image)
print(prediction)
predict_index = np.argmax(prediction)

if(predict_index==0):
    print("cat")
elif(predict_index==1):
    print("dog")
   
else:
    print("not know")