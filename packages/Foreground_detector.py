#!/usr/bin/python

####################################################################################
# File name : Foreground_detector.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - foreground_detector class 
#              - using cv2.grabCut method to extract foreground from background
#	       - Inputs are an image and the size of a box that absolutly the foreground located on it
# 	       - output is an image showing foreground		
#
####################################################################################
# import the necessary packages
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
####################################################################################

class foreground_detector(object):
  
  def foreground_detection(self, image, x, y, w, h):
      self.img = image
      self.mask = np.zeros(self.img.shape[:2], np.uint8)

      bgdModel = np.zeros((1,65), np.float64)
      fgdModel = np.zeros((1,65), np.float64)

      rect = (x, y, w, h)
      cv2.grabCut(self.img, self.mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
      mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
      self.img = self.img * mask2[:, :, np.newaxis]

      #cv2.imwrite("images/foreground.jpg", self.img)
      #plt.imshow(self.img)
      #plt.colorbar()
      #plt.show()
      return self.img
