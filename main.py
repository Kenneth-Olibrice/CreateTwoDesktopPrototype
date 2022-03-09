# Steps (for my reference):
# 1) Load the image files and their associated labels in parts.
# 2) For each part, format it into the desired format for the model (Dataset object I think).
# 3) Pass the data into the CNN for the feature map.
# 4) Pass the feature map to the DNN for processing
# 5) Test it on the test data, and evaluate on the eval data.
# 6) If the performance is acceptable, save the model and begin work on mobile importation.
import keras.layers
import pandas as pd
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from keras import layers, models, losses
from keras.layers import Dropout
import os

from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import adam_v2

# Define our variables/constants.
from keras_preprocessing.image import ImageDataGenerator

IMAGES_PATH = pathlib.Path("D:\\APMobileAppCreate#2\\Images")
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 450
batch_size = 32
SAVE_PATH = "D:\\APMobileAppCreate#2\\CheckpointDir\\cp.ckpt"
# We need to create an object to hold our labels.
meta_data = pd.read_csv("D:\\APMobileAppCreate#2\\HAM10000_metadata.csv")
label_list = list(meta_data["dx"])

# I reformatted the CSV file to change the 'dx' column to numerical instead of categorical. Here is the translation:
# 0) bkl
# 1) nv
# 2) df
# 3) mel
# 4) vasc
# 5) bcc
# 6) akiec

# Create dataset objects.


# End of dataset creation.
data_gen= ImageDataGenerator(validation_split=0.2)
train_ds = data_gen.flow_from_directory(IMAGES_PATH, class_mode="sparse",save_format='jpg',
                                     batch_size=batch_size, target_size=(45, 60),subset="training")

val_ds = data_gen.flow_from_directory(IMAGES_PATH, class_mode="sparse",save_format='jpg',
                                     batch_size=batch_size, target_size=(45, 60), subset="validation")

model = Sequential([
  keras.layers.Rescaling(1./255, input_shape=(45, 60, 3 )),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(7, activation="softmax")
])

model.compile(optimizer=keras.optimizers.adam_v2.Adam(),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(
  train_ds,
  validation_data=val_ds,
  shuffle=True,
  epochs=35,
  batch_size=batch_size,

)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

print(acc, "\n", val_acc)



