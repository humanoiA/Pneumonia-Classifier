import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
def predictor():
    loaded_model = load_model("./conv.h5")
    l=[]
    im=cv2.imread('uploads//'+'test.jpg',0)
    im_res=cv2.resize(im, (64, 64))
    a=img_to_array(im_res)
    a = a/255.0
    l.append(a)
    pred=np.array(l)
#print(df_test.iloc[0,1])
    out=loaded_model.predict(pred)
#print(out[0][0])
    pred_out=out[0][1]
    if(pred_out<0.5):
        return "Normal"
    else:
        return "Pneumonia"