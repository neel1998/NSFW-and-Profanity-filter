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

parser.add_argument("-t", "--train_dir", type=str, default="/ssd_scratch/cvit/anchit/train/", help=\
    "Path to the input training data.")
parser.add_argument("-b", "--batch_size", type=int, default=128, help= \
    "Batch size")
parser.add_argument("-e", "--epochs", type=int, default=10, help= \
    "Number of epochs")
parser.add_argument("--lr", type=float, default=0.0001, help= \
    "Learning rate")
parser.add_argument("-o", "--output_name", type=str, default="mobileNetV2.h5", help= \
    "name of saved model")
parser.add_argument("-v", "--verbose", type=int, default=1, help= \
    "Batch size")

args = parser.parse_args()

# Parameters
params = {'dim': (224,224),
		  'batch_size': args.batch_size,
		  'n_classes': 2,
		  'n_channels': 3,
		  'shuffle': True}

# Datasets
partition = {'train': [], 'validation': []} # IDs
labels = {'train': [], 'validation': []} # Labels

for TYPE in [['nsfw', 1], ['sfw', 0]]:

	listdir = os.listdir(join(args.train_dir, TYPE[0]))
	
	for j, i in enumerate(listdir):
		# Train validation 80-20 split
		if j < int(0.8 * len(listdir)):
			partition['train'].append(join(args.train_dir, TYPE[0], i))
			labels['train'].append(TYPE[1])
		else:
			partition['validation'].append(join(args.train_dir, TYPE[0], i))
			labels['validation'].append(TYPE[1])

if args.verbose:
	print(len(partition['train']), len(partition['validation']))

# Generators
training_generator = DataGenerator(partition['train'], labels['train'], **params)
validation_generator = DataGenerator(partition['validation'], labels['validation'], **params)

# Pre trained MobileNetV2
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

for layer in model.layers:
    layer.trainable=True    

if args.verbose:
	print(model.summary())

# Optimizer
optimizer_adam = optimizers.Adam(learning_rate=args.lr)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer_adam,
              metrics=['accuracy'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=args.output_name,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(training_generator,
                              epochs=args.epochs,
                              validation_data=validation_generator, 
                              callbacks=[model_checkpoint_callback],
                              verbose=args.verbose)