import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
from tqdm import tqdm
from keras.models import Model
from keras.applications.xception import *
from keras.preprocessing import image
from keras.utils import np_utils
from keras.layers import Dropout, Dense, Input, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np 
from scipy.misc import imread, imresize

from config import *


batch_size = 16
train_img = np.zeros([4750, img_size, img_size, 3])
train_label = np.zeros([4750, 1])

i = 0
for index, label in tqdm(enumerate(labels), total=len(labels)):
    for file in os.listdir('seg_train/' + label):
        im = imread('seg_train/{}/{}'.format(label, file))
        train_img[i,:,:,:] = imresize(im[:,:,:3], (img_size, img_size))
        train_label[i] = index
        i += 1

train_label = np_utils.to_categorical(train_label, 12)


datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             rotation_range=180,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             zoom_range=0.3,
                             horizontal_flip=True,
                             vertical_flip=True)

datagen.fit(train_img)


base_model = Xception(weights='imagenet', input_shape=(img_size, img_size, 3), include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='Adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(datagen.flow(train_img, train_label, batch_size=batch_size), steps_per_epoch=len(train_img)//batch_size, epochs=100, verbose=1)
model.save_weights('Xception.h5')