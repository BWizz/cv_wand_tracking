#!/usr/bin/env python3
# Script to train neural network model.
# Author: Brian Wisniewski
# Date: March 29th 2022
# CHANGELOG
# - Initial creation 
from cv2 import normalize
import tensorflow as tf
import cv2

#Import script from a different directory
import os
os.sys.path.insert(1, '../src')
from spell_detector_nn import spell_detector_model

train_ds = tf.keras.utils.image_dataset_from_directory(
    '../../training_data/',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(20,20),
    batch_size=35
)

spells = train_ds.class_names

val_ds = tf.keras.utils.image_dataset_from_directory(
    '../../training_data/',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(20,20),
    batch_size=35
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

checkpoint_path = "../src/spell_detector_model_parameters/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model = spell_detector_model(2)

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3,
  callbacks=[cp_callback]  # Pass callback to training
)

print(spells)
