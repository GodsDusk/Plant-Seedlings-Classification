import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
import numpy as np
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from config import *


datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             rotation_range=180,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             zoom_range=0.3,
                             horizontal_flip=True,
                             vertical_flip=True)


base_model = Xception(weights=None, include_top=False, input_shape=(img_size, img_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('Xception.h5')

with open('submission.csv', 'w') as f:
    f.write('file,species\n')
    for file in tqdm(os.listdir('seg_test/')):
        img = image.load_img(os.path.join('seg_test', file), target_size=(img_size, img_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.zeros([12,])
        for i, im in enumerate(datagen.flow(x)):
            pred += model.predict(im)[0]
            if i > 100:
                break
        f.write('{},{}\n'.format(file, labels[np.where(pred==np.max(pred))[0][0]]))
