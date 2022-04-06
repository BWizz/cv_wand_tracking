#!/usr/bin/env python3
# Prototype wand tip tracking algorithm
# Author: Brian Wisniewski
# Date: March 31th 2022
# CHANGELOG
# - Initial creation 
from cmath import inf
import os
#from os import XATTR_SIZE_MAX
import cv2
from cv2 import EVENT_FLAG_SHIFTKEY
import numpy as np
import time, sys
from signal import signal, SIGINT

def handler(signal_received, frame):

    print('TRL-C detected. Exiting gracefully')
    cv2.destroyAllWindows()
    sys.exit(0)

signal(SIGINT, handler)

class wandTipTracker():
    def __init__(self,width,height):
        self.state = "SEARCHING"
        self.lastMovementTime = 0
        self.lastTipPosition = (0,0)
        self.capturedWandPositions = []
        self.width = width
        self.height = height
        self.recordCaptures = True
        self.saveLocation = '../wandCaps/'

    #Normalizes a list of points (x,y) to a 16 x 16 grid
    def normalizePts(self,pts):
        x_max = max([pt[0] for pt in pts])
        x_min = min([pt[0] for pt in pts])
        y_max = max([pt[1] for pt in pts])
        y_min = min([pt[1] for pt in pts])

        out = []
        for pt in pts:
            new_x = int((pt[0] - x_min)/(x_max - x_min) * 16)
            new_y = int((pt[1] - y_min)/(y_max - y_min) * 16)
            out.append((new_x,new_y))
        return out
    
    def searching_state(self,distanceTraveled):
        print(distanceTraveled)
        self.capturedWandPositions = []
        if self.lastMovementTime == 0:
            if distanceTraveled <= 2:
                self.lastMovementTime = time.time()
        elif distanceTraveled > 10:
            self.lastMovementTime = 0
        elif (time.time() - self.lastMovementTime) > 1:
            self.state = "WAITING"

    def waiting_state(self,distanceTraveled,currentTipPosition):
        if (time.time() - self.lastMovementTime) > 10:
            self.state = "SEARCHING"
        elif distanceTraveled > 10:
            self.capturedWandPositions.append(self.lastTipPosition)
            self.capturedWandPositions.append(currentTipPosition)
            self.lastMovementTime = 0
            self.state = "TRACKING"

    def tracking_state(self,distanceTraveled,currentTipPosition):
        if distanceTraveled > 1:
            self.capturedWandPositions.append(currentTipPosition)
        elif self.lastMovementTime == 0:
            self.lastMovementTime = time.time()
        elif ( time.time() - self.lastMovementTime ) > 1:
            self.lastMovementTime = 0
            self.state = "VALIDATING"

    def validating_state(self):
        numPoints = len(self.capturedWandPositions)
        newPts = self.normalizePts(self.capturedWandPositions)
        print(newPts)
        blankImg = np.zeros((20,20,3), np.uint8)
        for idx in range(numPoints):
            if idx < numPoints-1:
                capturedImage = cv2.line(blankImg, newPts[idx], newPts[idx+1], (255, 0, 0), 1)
        self.state = "SEARCHING"

        #For testing only
        cv2.imshow("Captured", capturedImage)

        if self.recordCaptures:
            now = str(time.time())
            cv2.imwrite(self.saveLocation + '_' + now + '.png',capturedImage)

        time.sleep(2)

    def run(self,frame):
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
        thresh = cv2.threshold(grayImage, 250, 255, cv2.THRESH_BINARY)[1]
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(thresh)

        #For testing only
        cv2.circle(grayImage, maxLoc, 10, (255, 0, 0), 2)
        cv2.imshow("Original", grayImage)
        cv2.imshow("processed", thresh)

        # Create folder if it does not already exist
        if self.recordCaptures  and not os.path.exists(self.saveLocation):
            os.makedirs(self.saveLocation)
        
        if maxLoc[0] != 0 and maxLoc[1] != 0:

            #Calculate distance traveled
            xdiff = (maxLoc[0] - self.lastTipPosition[0])**2
            ydiff = (maxLoc[1] - self.lastTipPosition[1])**2
            dist = np.sqrt(xdiff+ydiff)

            if self.state == "SEARCHING":
                self.searching_state(dist)
            elif self.state == "WAITING":
                self.waiting_state(dist, maxLoc)
            elif self.state == "TRACKING":
                self.tracking_state(dist, maxLoc)
            elif self.state == "VALIDATING":
                self.validating_state()
            else:
                print("Error, should never get here")

            self.lastTipPosition = maxLoc
            
            #For testing only
            print(self.state)

cap = cv2.VideoCapture(0)
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
tracker = wandTipTracker(width,height)

while True:
    success, img = cap.read()
    tracker.run(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break