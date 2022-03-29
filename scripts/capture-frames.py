#!/usr/bin/env python3
# Simple python script to record video frames for computer vision testing
# Author: Brian Wisniewski
# Date: March 29th 2022
# CHANGELOG
# - Initial creation 
import argparse
import cv2
import time
import os

parser = argparse.ArgumentParser(description='Record camera frames to local folder.')
parser.add_argument('--duration', dest='duration', type=int, default=10, help='Recording length in seconds')
parser.add_argument('--delay', dest='delay', type=int, default=5, help='Recording delay in seconds')
parser.add_argument('--path', dest='path', type=str, default='recordings', help='Path to location to save frames')
args = parser.parse_args()

start_time = time.time()
save_location = args.path + "_" + str(start_time)
save_location = save_location.replace(".","_")

cap = cv2.VideoCapture(0)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create folder if it does not already exist
if not os.path.exists(args.path):
    os.makedirs(args.path)

while (time.time() - start_time) < (args.duration + args.delay):
    success, img = cap.read()
    if success:
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        file_name = args.path + '/test_frame_' + str(time.time())+'.png'

        if cv2.imwrite(file_name, grayImage):
            print(file_name + ": Saved Successfully")
        else:
            # If a frame fails to save, it could be because the location
            # does not exist, or the permissions of the location.
            print(file_name + ": Failed to Save!")