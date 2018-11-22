#!/usr/bin/python

####################################################################################
# File name : Image_functions.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - image_functions class
#              - reading, resizing, converting, thresholding and .... image functions	
#
####################################################################################
# import the necessary packages
import cv2
import imutils
import numpy as np

####################################################################################
class image_functions:
  #def __init__(self):
 
  # read image file
  def read_image(self, filename):
      image = cv2.imread(filename) # Read an image
      return image

  # resize image file with the new size
  def do_resize(self, image, new_size):
      resized = imutils.resize(image, width=new_size)
      ratio = image.shape[0] / float(resized.shape[0])
      return (ratio, resized)
    
  # Converts an RGB image to grayscale, where each pixel
  # now represents the intensity of the original image.
  def rgb_to_gray(self, image):
      image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      return image_gray

  # Converts an RGB image to HSV format
  def rgb_to_hsv(self, image):
      image_blur = cv2.blur(image,(3,3))
      image_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
      return image_hsv

  # Doing the Gaussian Blur
  def do_GaussianBlur(self, image, blur_size):
      image_gaussianBlur = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
      return image_gaussianBlur

  # Converts an image into a binary image at the specified threshold and a specific thresholding type.
  # All pixels with a value <= threshold become 0, while pixels > threshold become 1
  # 1 means simple binary, 2 means inverse binary, 3 means Trunc, 4 means Tozero, 5 means inverse Tozero, 6 means Otsu's Binarization
  def do_threshold(self, image, threshold, thresholing_type):
      if thresholing_type == 1:
         ret, img_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
      if thresholing_type == 2:
         ret, img_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
      if thresholing_type == 3:
         ret, img_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_TRUNC)
      if thresholing_type == 4:
         ret, img_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO)
      if thresholing_type == 5:
         ret, img_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO_INV) 
      if thresholing_type == 6:      
         ret, img_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
      return (ret, img_thresh)

  # Adaptive Thresholding mean: threshold value is the mean of neighbourhood area.
  # Block Size - It decides the size of neighbourhood area.
  # C - It is just a constant which is subtracted from the mean or weighted mean calculated.
  def do_adaptiveThreshold_mean(self, image, Block_size, C):
      image_adaptiveThreshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, Block_size, C)
      return image_adaptiveThreshold

  # Adaptive Thresholding Gaussian: threshold value is the weighted sum of neighbourhood values 
  # where weights are a gaussian window.
  # Block Size - It decides the size of neighbourhood area.
  # C - It is just a constant which is subtracted from the mean or weighted mean calculated.
  def do_adaptiveThreshold_Gaussian(self, image, Block_size, C):
      image_adaptiveThreshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, Block_size, C)
      return image_adaptiveThreshold

  # Morphological Transformations
  # 1 means erosion, 2 means dilation, 3 means opening, 4 means closing, 5 means morphology gradient
  def do_morphologyEx(self, image, kernel_size, operation_type):
      kernel = np.ones((kernel_size, kernel_size), np.uint8)
      if operation_type == 1:
         img_morphology = cv2.erode(image, kernel, iterations = 1)
      if operation_type == 2:
         img_morphology = cv2.dilate(image, kernel, iterations = 1)
      if operation_type == 3:
         img_morphology = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
      if operation_type == 4:
         img_morphology = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
      if operation_type == 5:
         img_morphology = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)   
      return img_morphology 

  # Resize it to a smaller factor so that the shapes can be approximated better
  # convert image to binary image
  def convert_to_binary(self, image):
      self.img = image
      (self.ratio, self.resized) = self.do_resize(self.img, self.img.shape[0])
      
      # convert the resized image to grayscale, blur it slightly,
      # and threshold it
      self.img_gray = self.rgb_to_gray(self.resized) # Convert img from RGB to Grayscale
      self.img_blured = self.do_GaussianBlur(self.img_gray, 5) # blur img 
      
      (self.thresh, self.img_threshold) = self.do_threshold(self.img_blured, 80, 4)
      
      self.img_threshold_mean = self.do_adaptiveThreshold_mean(self.img_threshold, 11, 2)
      self.img_threshold_Gaussian = self.do_adaptiveThreshold_Gaussian(self.img_threshold, 11, 2)
      
      # 1 means erosion, 2 means dilation, 3 means opening, 4 means closing, 5 means morphology gradient
      self.img_threshold_morph = self.do_morphologyEx(self.img_threshold, 3, 2)

      #cv2.imwrite("images/Threshold_image.jpg", self.img_threshold)
      #cv2.imwrite("images/Threshold_Mean_image.jpg", self.img_threshold_mean)
      #cv2.imwrite("images/Threshold_Gaussian_image.jpg", self.img_threshold_Gaussian)
      cv2.imwrite("images/Threshold_Morph_image.jpg", self.img_threshold_morph) 
     
      return self.img_threshold_morph


  # convert image to HSV format
  def convert_to_hsv(self, image):
      self.img = image
      #(self.ratio, self.resized) = self.do_resize(self.img, self.img.shape[0])
      # convert the resized image to HSV
      self.img_hsv = self.rgb_to_hsv (self.img)
      cv2.imwrite("images/HSV_image.jpg", self.img_hsv)
      return self.img_hsv

  # Canny edge detection
  def canny(self, binary_image):
      # canny edges detection function
      canny_edges = cv2.Canny(binary_image, 150, 250)
      canny_edges = cv2.convertScaleAbs(canny_edges)
      cv2.imwrite("images/Canny_edges_image.jpg", canny_edges)
      return canny_edges
