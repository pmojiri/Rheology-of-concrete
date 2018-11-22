#!/usr/bin/python

####################################################################################
# File name : Object_detector.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - object_detector class
#              - functions are defined for object detection	
#
####################################################################################
#
# import the necessary packages
import cv2
import imutils
import numpy as np
import sys
sys.path.append("/home/viki/Desktop/Slump_test/packages")

from Image_functions import image_functions

################################################################################

class object_detector:
  def __init__(self):
      self.image = image_functions()

  # Object detection using thresholing (simple or Otsu)
  def Obj_gray_detection(self, image, th1, th2):
      self.img = image
      self.th_lower = th1
      self.th_upper = th2
      #(self.ratio, self.resized) = self.image.do_resize(self.img, self.img.shape[0])
      
      # convert the resized image to grayscale, blur it slightly,
      # and threshold it
      self.img_gray = self.image.rgb_to_gray(self.img) # Convert img from RGB to Grayscale
      self.img_gray = cv2.equalizeHist(self.img_gray)
      self.img_blured = self.image.do_GaussianBlur(self.img_gray, 5) # blur img to remove noises

      # define range of gray
      lower_gray = np.array([self.th_lower], dtype = np.uint8)
      upper_gray = np.array([self.th_upper], dtype = np.uint8)

      self.gray_filtered = cv2.inRange(self.img_blured, lower_gray, upper_gray)
      # 1 means erosion, 2 means dilation, 3 means opening, 4 means closing, 5 means morphology gradient
      self.gray_filtered = self.image.do_morphologyEx(self.gray_filtered, 3, 2)
           
      (self.thresh, self.img_threshold) = self.image.do_threshold(self.img_blured, self.th_lower, 4)
      (self.thresh, self.gray_detected) = self.image.do_threshold(self.img_threshold, self.th_upper, 5)     
      # 1 means erosion, 2 means dilation, 3 means opening, 4 means closing, 5 means morphology gradient
      self.gray_detected = self.image.do_morphologyEx(self.gray_detected, 3, 2)
      #self.gray_res = cv2.bitwise_and(self.img, self.img, mask = self.gray_detected)


      cv2.imwrite("images/Gray_detected_image.jpg", self.gray_detected)
      cv2.imwrite("images/Gray_filtered_image.jpg", self.gray_filtered) 
      #
 
      return (self.gray_detected, self.gray_filtered)

  # Object detection using HSV colors
  def Obj_color_detection(self, image, lower, upper):
      self.img = image
      self.img_hsv = self.image.convert_to_hsv(self.img)
     
      # define range of concrete color in HSV
      lower_concrete = np.array(lower, dtype = np.uint8)
      upper_concrete = np.array(upper, dtype = np.uint8)

      self.mask_concrete = cv2.inRange (self.img_hsv, lower_concrete, upper_concrete)
      # 1 means erosion, 2 means dilation, 3 means opening, 4 means closing, 5 means morphology gradient
      #self.mask_concrete = self.image.do_morphologyEx(self.mask_concrete, 3, 2)
      self.color_res = cv2.bitwise_and(self.img, self.img, mask = self.mask_concrete)

      cv2.imwrite("images/Masked_image.jpg", self.mask_concrete)
      cv2.imwrite("images/Res.jpg", self.color_res) 
      return (self.mask_concrete, self.color_res)


  # Find contours
  def find_obj_contours(self, image):
      self.obj_mask = image
      self.contours, self.hierarchy = cv2.findContours(self.obj_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      return (self.contours, self.hierarchy)




