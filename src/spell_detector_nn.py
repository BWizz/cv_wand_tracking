#!/usr/bin/env python3
# Python module of a tensorflow neural network model 
# designed for character detection.
#
# Author: Brian Wisniewski
# Date: March 29th 2022
# CHANGELOG
# - Initial creation 
import tensorflow as tf

def spell_detector_model(num_spells):

  #Character detection model
  model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(input_shape=(20,20)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(num_spells)
  ])

  model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])
  return model

def load_spell_detector(path,num_spells):
  model = spell_detector_model(num_spells)
  model.built = True
  model.load_weights(path)
  return model
