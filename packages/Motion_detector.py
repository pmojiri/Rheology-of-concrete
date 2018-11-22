#!/usr/bin/python

####################################################################################
# File name : Motion_detector.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - motion_detector class	
#
####################################################################################
#
# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
####################################################################################
class motion_detector:
  def __init__(self):
      # initialize the first frame in the video stream
      self.firstFrame = None
  
  def motion_detection(self, video, min_area):
      self.camera = video
      self.min_area = min_area
      #i = 0
      # loop over the frames of the video
      while True:
	      # grab the current frame and initialize the occupied/unoccupied
	      # text
	      (self.grabbed, self.frame) = self.camera.read()
	      text = "Unoccupied"
              
              #cv2.imwrite("image" + str(i) + ".jpg", self.frame)
              #i = i +1
	      
              # if the frame could not be grabbed, then we have reached the end
	      # of the video
	      if not self.grabbed:
		    break

	      # resize the frame, convert it to grayscale, and blur it
	      self.frame = imutils.resize(self.frame, width=500)
	      self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
	      self.gray = cv2.GaussianBlur(self.gray, (21, 21), 0)

	      # if the first frame is None, initialize it
	      if self.firstFrame is None:
	            self.firstFrame = self.gray
		    continue

	      # compute the absolute difference between the current frame and
	      # first frame
	      self.frameDelta = cv2.absdiff(self.firstFrame, self.gray)
	      self.thresh = cv2.threshold(self.frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	      # dilate the thresholded image to fill in holes, then find contours
	      # on thresholded image
	      self.thresh = cv2.dilate(self.thresh, None, iterations=2)
	      (cnts, _) = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	      # loop over the contours
	      for c in cnts:
	            # if the contour is too small, ignore it
		      if cv2.contourArea(c) < self.min_area:
		            continue

		      # compute the bounding box for the contour, draw it on the frame,
		      # and update the text
		      (x, y, w, h) = cv2.boundingRect(c)
		      cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		      text = "Occupied"

	      # draw the text and timestamp on the frame
	      cv2.putText(self.frame, "Room Status: {}".format(text), (10, 20),
		      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	      cv2.putText(self.frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		      (10, self.frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	      # show the frame and record if the user presses a key
	      cv2.imshow("Security Feed", self.frame)
	      cv2.imshow("Thresh", self.thresh)
	      cv2.imshow("Frame Delta", self.frameDelta)
	      key = cv2.waitKey(1) & 0xFF

	      # if the `q` key is pressed, break from the lop
	      if key == ord("q"):
		      break

      # cleanup the camera and close any open windows
      self.camera.release()
      cv2.destroyAllWindows()
