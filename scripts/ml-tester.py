#!/usr/bin/env python3
# Script to test neural network model
# Author: Brian Wisniewski
# Date: March 29th 2022
# CHANGELOG
# - Initial creation 
import tensorflow as tf
import numpy as np
import cv2

#Import script from a different directory
import os
os.sys.path.insert(1, '../src')
from spell_detector_nn import load_spell_detector

model = load_spell_detector("../src/spell_detector_model_parameters/cp.ckpt",2)
test_files = ['../exampleCaptures/Bad-Spell/bad-spell.png', \
            '../exampleCaptures/Lumos/lumos.png', \
            '../exampleCaptures/Wingardium-Leviosa/wl.png']

#test_files = [os.path.join('..', 'exampleCaptures', 'Bad-Spell', 'bad-spell.png')]

for file in test_files:
    im = cv2.imread(file)
    cv2.imshow('Test',im)
    img_array = tf.keras.utils.img_to_array(im)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    max_score = 100 * np.max(score)

    print('=========================================')
    if (max_score > 80):
        class_names = ['Luminos', 'Wingardium Leviosa']
        outPut = "DETECTED " + class_names[np.argmax(score)] + \
            " : " + str(float(int(max_score*100)/100)) 
        print(outPut)
    else:
        print("FAILED TO DETECT SPELL")
    print('=========================================')