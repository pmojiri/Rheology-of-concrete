#!/usr/bin/python

####################################################################################
# File name : Height_detector.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - Height_detector class
#              - using simple color detection to detect the concrete area and computing the effected area
#		and using cone transformation matrix the real concrete height are computed
#	       - Inputs are an image and the cropped area 
# 	       - outputs are the height of detected area and an image showing the extracted area	
#
####################################################################################

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import yaml
import json
import numpy as np
from matplotlib import pyplot as plt
import argparse
from skimage.filters import threshold_otsu, threshold_adaptive

import sys
sys.path.append("/home/viki/Desktop/Slump_test/packages")

from Image_functions import image_functions
from Video_functions import video_functions
from Object_detector import object_detector
from Entropy import Entropy

from skimage.morphology import disk
from skimage.filters import threshold_otsu

################################################################################
class height_detector:
  def __init__(self):
      	# initialize the first frame in the video stream
	self.image = image_functions()
	self.object_detection = object_detector()
	self.entropy = Entropy()
  
  def height_detection(self, image, x_top, x_bottom):
    	source_color = image
	source = image
	source_2 = image
        x1 = x_top
        x2 = x_bottom
	#cv2.imshow("Original image", source_color)
	#cv2.waitKey(0)

	mask = np.zeros((source_color.shape[0], source_color.shape[1], 3), np.uint8)
	mask[x1: x2, 0:900] = image[x1: x2, 0:900]
	#cv2.imshow("Cropped", mask)
	#cv2.waitKey(0)

	source_color = mask
	#cv2.imshow("original image", source_color)
	#cv2.waitKey(0)

	with open('files/transformationMatrix_cone.yaml', "r") as f:
    		doc = yaml.load(f)

	#print doc
	transformationMatrix = np.asarray(doc["homography_matrix"])

	# Object detection using HSV range color detector
	hsv_img = cv2.cvtColor(source_color, cv2.COLOR_BGR2HSV)
	#COLOR_MIN = np.array([10, 0, 100],np.uint8) 
	#COLOR_MAX = np.array([30, 100, 200],np.uint8)
	COLOR_MIN = np.array([10, 0, 100],np.uint8) 
	COLOR_MAX = np.array([20, 100, 130],np.uint8)
	obj_mask = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
	kernel = np.ones((3, 3), np.uint8)
	obj_mask = cv2.dilate(obj_mask, kernel, None)
	cv2.imwrite("images/result_images/Masked height area.jpg", obj_mask)
	contours_c, hierarchy_c = cv2.findContours(obj_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(obj_mask, contours_c, -1, (255,0,0), 1)
	x, y, w, h = cv2.boundingRect(max(contours_c, key = cv2.contourArea))
	cv2.rectangle(source_color, (x, y), (x + w, y + h), (255, 0, 0), 1)
	#cv2.imshow("masked", obj_mask)
	#cv2.waitKey(0)

	p1 = np.array([[(x + w/2)], [y], [1]])
	p2 = np.array([[(x + w/2)], [(y + h)], [1]])
	cv2.circle(source_color, (p1[0, 0], p1[1][0]), 10, (0, 255, 0), 2)
	cv2.circle(source_color, (p2[0, 0], p2[1][0]), 10, (0, 255, 0), 2)

	cont, hier = cv2.findContours(obj_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	c_m = max(cont, key = cv2.contourArea)
	cv2.drawContours(source_color, [c_m], 0, (255,0, 255), 1)

	#cv2.imwrite("images/result_images/Detected_height_area.jpg", source_color)
	#cv2.imshow("Detected height area", source_color)
	#cv2.waitKey(0)

	# computing real value using image points and cone transformation matrix
	P1 = np.dot(transformationMatrix, p1)
	P2 = np.dot(transformationMatrix, p2)

	P_top = (P1[0, 0] / P1[2, 0], P1[1, 0] / P1[2, 0])
	P_bottom = (P2[0, 0] / P2[2, 0], P2[1, 0] / P2[2, 0])

	height = np.sqrt(np.power((P_bottom[0] - P_top[0]),2) + np.power((P_bottom[1] - P_top[1]),2))
	print "concrete height is : " + str(height)

	return height, source_color

