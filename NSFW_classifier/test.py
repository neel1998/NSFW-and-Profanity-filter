#imports

import numpy as np
import os
from os.path import join
import cv2
import argparse

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers

from dataGenerator import DataGenerator

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", type=str, required=True, help=\
    "Image path to be predicted")
parser.add_argument("-c", "--ckpt", type=str, default='mobileNetV2.h5', help= \
    "Checkpoint path")

args = parser.parse_args()

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(1,activation='sigmoid')(x)

model=Model(inputs=base_model.input,outputs=preds)

model.load_weights(args.ckpt)

optimizer_adam = optimizers.Adam(learning_rate=1e-4)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer_adam,
              metrics=['accuracy'])

img = cv2.resize(cv2.imread(args.image), (224, 224))

img = img.reshape((1, 224, 224, 3))
img = tf.convert_to_tensor(img)

out = model.predict(img)
print("Score:", out)