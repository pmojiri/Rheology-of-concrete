#!/usr/bin/python

####################################################################################
# File name : Radius_detector.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - radius_detector class
#              - using simple colour detection to detect the concrete area and computing the effected area
#		and export the distance of the extracted edges from centre point
#	       - Inputs are an image, centre point and top point
# 	       - outputs are the radius of detected area and an image showing the extracted area
#		and width and height of the fitted rectangle to extracted area	
#
####################################################################################
# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
from skimage.filters import threshold_otsu, threshold_adaptive

import sys
sys.path.append("/home/viki/Desktop/Slump_test/packages")

from Image_functions import image_functions
from Video_functions import video_functions
from Object_detector import object_detector
from Click_and_Crop import click_and_crop
from Motion_detector import motion_detector
from Entropy import Entropy

from skimage.morphology import disk
from skimage.filters import threshold_otsu


################################################################################
class radius_detector:
  def __init__(self):
      	# initialize the first frame in the video stream
	self.image = image_functions()
	self.object_detection = object_detector()
	self.entropy = Entropy()
  
  def radius_detection(self, image, cx, cy, y_top):
    	image_original = image
        image_original = cv2.GaussianBlur(image_original, (15, 15), 0)
        #image_original = cv2.GaussianBlur(image_original, (11, 11), 0)

    	# Object detection using HSV range color detector
    	hsv_img = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    	
	#COLOR_MIN = np.array([5, 0, 100],np.uint8)     	
	#COLOR_MAX = np.array([20, 100, 200],np.uint8)

	COLOR_MIN = np.array([5, 40, 0],np.uint8) 
	COLOR_MAX = np.array([20, 100, 200],np.uint8)

	#COLOR_MIN = np.array([5, 50, 30],np.uint8) 
	#COLOR_MAX = np.array([20, 100, 180],np.uint8)

    	obj_mask = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
    	kernel = np.ones((9, 9), np.uint8)
    	#kernel = np.ones((3, 3), np.uint8)
    	#obj_mask = cv2.dilate(obj_mask, kernel, None)
    	obj_mask = cv2.erode(obj_mask, kernel, None)
    	#cv2.imwrite("images/result_images/Masked.jpg", obj_mask)
    	#cv2.imshow("masked", obj_mask)
    	#cv2.waitKey(0)

    	sigma = 0.33
    	v = np.median(image_original)
    	lower = int(max(0, (1.0 - sigma) * v))
    	upper = int(min(255, (1.0 + sigma) * v))
    	#lower = threshold_otsu(image_original)
    	#upper = threshold_otsu(image_original) * 0.5
    	edges = cv2.Canny(image_original, lower, upper, apertureSize = 3)
    	edges = cv2.convertScaleAbs(edges)
    	edges = cv2.GaussianBlur(edges, (11, 11), 0)    
    	#cv2.imwrite("images/result_images/scc_edges.jpg", edges)
    	#cv2.imshow("edges", edges)
    	#cv2.waitKey(0)

    	#obj_mask = cv2.bitwise_and(obj_mask, obj_mask, mask = edges)
    	#cv2.imwrite("images/result_images/scc_edges_2.jpg", edges)
    	#cv2.imshow("new edges", edges)
    	#cv2.waitKey(0)

    	#image_entropy = self.entropy.Obj_entropy_detection(image_original, 3)
    	#obj_mask = cv2.bitwise_and(image_entropy, image_entropy, mask = obj_mask)
    	#kernel = np.ones((5, 5), np.uint8)
    	#obj_mask = cv2.dilate(edges, kernel, None)
    	#edges = cv2.erode(edges, kernel, None)
    	#cv2.imwrite("images/result_images/scc_edges_3.jpg", edges)
    	#cv2.imshow("new edges", edges)
    	#cv2.waitKey(0)

    	contours_c, hierarchy_c = self.object_detection.find_obj_contours(obj_mask)
    	cv2.drawContours(image_original, [max(contours_c, key = cv2.contourArea)], -1, (255,255,0), 1)
	#cv2.drawContours(image_original, contours_c , -1, (255,255,0), 1)
	x, y, w, h = cv2.boundingRect(max(contours_c, key = cv2.contourArea))
	cv2.rectangle(image_original, (x, y), (x + w, y + h), (255, 0, 0), 1)
        print "width  is:" + str(w)
        print "height is:" + str(h)

    	c_m = max(contours_c, key = cv2.contourArea)
    	M = cv2.moments(c_m)
    	#cx = int(M['m10']/M['m00'])
    	#cy = int(M['m01']/M['m00'])
    	#print cx, cy
    	cv2.circle(image_original, (cx, cy), 5, (255, 0, 0), 3)
 
    	c_m = c_m.tolist()
        #print "c_m is:" + str(c_m)

    	radius = []
    	d = []
        index = []
    	for m in xrange(0, len(c_m)):
                #print c_m
                x = c_m[m][0][0]
                y = c_m[m][0][1] 
       		if y <= y_top + 20:
          		index.append(m)
	c_m_new = np.delete(c_m, (index), axis = 0)

    	for n in xrange(0, len(c_m_new)):
      		x = c_m_new[n][0][0]
       		y = c_m_new[n][0][1] 
       		dist = np.sqrt(np.power((x - cx),2) + np.power((y - cy),2))
       		radius.append(dist)

    	cv2.drawContours(image_original, [c_m_new], 0, (0,255, 0), 2)
    	#cv2.imwrite("images/result_images/filtered_area.jpg", image_original)
    	#cv2.imshow("Detected area", image_original)
   	#cv2.waitKey(0)

    	#print "radius are" + str(radius)
    	#print "max radius:" + str(max(radius))
    	#print "min radius:" + str(min(radius))
    	print "average radius:" + str(reduce(lambda x, y: x + y, radius) / len(radius))
    	for i in range(0, len(c_m_new)):
       		cv2.circle(image_original, (c_m_new[i][0][0], c_m_new[i][0][1]), 1, (255, 255, 255), 2)
    	#cv2.imwrite("images/result_images/filtered_area.jpg", image_original)
    	#cv2.imshow("Detected area", image_original)
    	#cv2.waitKey(0)
        return radius, image_original, w, h

