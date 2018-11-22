#!/usr/bin/python

####################################################################################
# File name : Video_functions.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - video_functions class
#              - reading and ... video function	
#
####################################################################################
# import the necessary packages
import cv2
import imutils
import numpy as np

class video_functions:
  #def __init__(self):
 
  # read video file
  def read_video(self, filename):
      camera = cv2.VideoCapture(filename)
      return camera


