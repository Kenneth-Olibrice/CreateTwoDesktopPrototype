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
import os
from keras.models import Sequential
from keras.optimizers import adam_v2

# Define our variables/constants.
IMAGES_PATH = pathlib.Path("D:\\APMobileAppCreate#2")
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 450
batch_size = 32
SAVE_PATH = "D:\\APMobileAppCreate#2\\CheckpointDir\\cp.ckpt"
SAVE_DIR = os.path.dirname(SAVE_PATH)
class_names = ["benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses",
               "melanocytic nevi", "dermatofibroma", "melanoma",
               "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage",
               "basal cell carcinoma", "carcinoma / Bowen's disease"]

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
train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=IMAGES_PATH, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                        labels=label_list, batch_size=batch_size, seed=123,
                                                        validation_split=0.2, subset="training")

val_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=IMAGES_PATH, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                        labels=label_list, batch_size=batch_size, seed=123,
                                                        validation_split=0.2, subset="validation")

# End of dataset creation.


model = Sequential([
  keras.layers.Rescaling(1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
  layers.Conv2D(32, 3, padding='same', activation='sigmoid'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='sigmoid'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='sigmoid'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='sigmoid'),
  layers.Dense(7)
])


model.compile(optimizer=keras.optimizers.sgd_experimental.SGD(),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(
  train_ds,
  shuffle=True,
  validation_data=val_ds,
  epochs=10,
  batch_size=batch_size,
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

print(acc, "\n", val_acc)

