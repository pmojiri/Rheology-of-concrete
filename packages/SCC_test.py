#!/usr/bin/python

####################################################################################
# File name : SCC_test.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: SCC test class
#              inputs: - an image directory or a video directory
#		       - plate size in milimeter
#		       - plate color		
#              output: - Excel file showing the requested data, the saved videos
####################################################################################
# import the necessary packages
import cv2
import numpy as np
import imutils
import argparse
import datetime
import time
import xlwt
from matplotlib import pyplot as plt
from imutils.video import VideoStream
import sys
sys.path.append("/home/viki/Desktop/Slump_test/packages")

from perspective import Perspective
from perspective import Perspective_plate
from perspective_cone import Perspective
from perspective_cone import Perspective_cone

from Image_functions import image_functions
from Video_functions import video_functions
from Foreground_detector import foreground_detector
from Radius_detector import radius_detector
from Object_detector import object_detector
#
################################################################################

class SCC_Test(object):
    def __init__(self, filepath, platesize, platecolor_min, platecolor_max):
        self.filename = filepath
        self.plate_size = int(platesize) #plate square size in milimeter
	self.plate_color_min = platecolor_min
	self.plate_color_max = platecolor_max

	# initialization
	self.image = image_functions()
	self.camera = video_functions()
	self.foreground = foreground_detector()
	self.radius = radius_detector()
	self.object_detection = object_detector()

	self.filename_image = 'excel_files/SCC_data_image.xls'
	self.filename_video = 'excel_files/SCC_data_video.xls'
	self.ezxf = xlwt.easyxf
	self.hdngs = ['  Time (s)   ', '   edge distance from center point   ',     '    Diameter_ew (mm)   ', '     Diameter_ns (mm)     ']
	self.heading_xf = self.ezxf('font: bold on; align: wrap on, vert centre, horiz center')
	self.data = []


	# write data on Excel file
    def write_xls(self, file_name, sheet_name, headings, data, heading_xf):

    	book = xlwt.Workbook()
    	sheet = book.add_sheet(sheet_name)
    	rowx = 0
    	for colx, value in enumerate(headings):
        	sheet.write(rowx, colx, value, heading_xf)
    	sheet.set_panes_frozen(True) # frozen headings instead of split panes
    	sheet.set_horz_split_pos(rowx+1) # in general, freeze after last heading row
    	sheet.set_remove_splits(True) # if user does unfreeze, don't leave a split there
    	for row in data:
        	rowx += 1
        	for colx, value in enumerate(row):
            		sheet.write(rowx, colx, value)
    	book.save(file_name)

	#we are reading image file (The image file should be the final image detecting four corners of the plate 
	#and the slipped concrete:	
    def scc_image(self):
    	# reading the inputs:
    	image = self.image.read_image(self.filename)
    	plate_size = self.plate_size

    	# computing trasformation matrix calling houghlineP method (plate perspective) and warped image (Top-down view of the plate)
    	persp = Perspective_plate(image, plate_size, self.plate_color_min, self.plate_color_max)
    	warpedImage, transformationMatrix = persp.houghlinePmethod()

    	# extracting the foreground from background to have a better concrete detetction (manually defined a box)
    	warpedImage_cropped = self.foreground.foreground_detection(warpedImage, (plate_size/2 - 350), (plate_size/2 - 350), (plate_size/2 + 200), (plate_size/2 + 200))

    	# concrete detection
    	cx = warpedImage.shape[1]/2
    	cy = warpedImage.shape[0]/2
    	r, detected_area, d_ew, r_ns = self.radius.radius_detection(warpedImage_cropped, cx, cy, (plate_size/2 - 350) )
    	#cv2.imshow('detected', detected_area)
    	#cv2.waitKey(0)
    
    	# save the detected area and write the data on a excel file
    	cv2.imwrite("images/result_images/SCC_detected.jpg", detected_area)
    	for i in xrange(0, len(r), 20):
		self.data.append([0, r[i], d_ew, r_ns])
    	self.write_xls(self.filename_image, "SCC_image", self.hdngs, self.data, self.heading_xf)
    
    	# showing the results
    	fig, axes  = plt.subplots(ncols=2, nrows =2)
    	ax0, ax1, ax2, ax3 = axes.ravel()

    	img0 = ax0.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    	ax0.set_title("Input image")
    	ax0.axis("off")

    	img1 = ax1.imshow(cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB))
    	ax1.set_title("Warped image")
    	ax1.axis("off")

    	img2 = ax2.imshow(cv2.cvtColor(warpedImage_cropped, cv2.COLOR_BGR2RGB))
    	ax2.set_title("Foreground image")
    	ax2.axis("off")

    	img3 = ax3.imshow(cv2.cvtColor(detected_area, cv2.COLOR_BGR2RGB))
    	ax3.set_title("Detected Area")
    	ax3.axis("off")

    	fig.tight_layout()
    	#plt.show()

        return d_ew


	# otherwise, we are reading from a video:
    def scc_video(self):
    	# reading the inputs:    
    	camera = self.camera.read_video(self.filename)
    	plate_size = self.plate_size
 
    	# Grab first frame of the video 
    	(first_grabbed, first_frame) = camera.read()
    	pos_frame = camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    	pos_time = camera.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    	print pos_frame, pos_time
    	cv2.imwrite("images/result_images/SCC_first_frame.jpg", first_frame)

    	# computing trasformation matrix using first frame of the video calling houghlineP method (plate perspective) and warped image (Top-down view of the plate)
    	persp = Perspective_plate(first_frame, plate_size, self.plate_color_min, self.plate_color_max)
    	warpedImage, transformationMatrix = persp.houghlinePmethod()

    	# computing trasformation matrix of cone (cone perspective) to compute center of cone (cone center as the refrence of radius extraction)
    	persp_cone = Perspective_cone(first_frame, 3000)
    	warpedImage_cone, transformationMatrix_cone, cone_height, x_top, x_bottom, bottom_point = persp_cone.cone_detection()

    	p1 = bottom_point
    	P1 = np.dot(transformationMatrix, p1)
    	cone_bottom = (int(P1[0, 0] / P1[2, 0]), int(P1[1, 0] / P1[2, 0]))
    	print "p1 is : " + str(p1)
    	print "P1 is : " + str(P1)
    	print "cone_bottom is : " + str(cone_bottom) 
    	cx = cone_bottom[0]
    	cy = cone_bottom[1]
    	print "cx is: " + str(cx)
    	print "cy is: " + str(cy)


    	# Define the codec and create VideoWriter object to save the detected area video
    	fourcc = cv2.cv.CV_FOURCC(*'XVID')
    	video_out = cv2.VideoWriter("videos/SCC_warped_plate.avi", fourcc, 30, (plate_size, plate_size))
    	video_cropped = cv2.VideoWriter("videos/SCC_cropped_plate.avi", fourcc, 30, (plate_size, plate_size))
    	video_detected = cv2.VideoWriter("videos/SCC_radius_detected.avi", fourcc, 30, (plate_size, plate_size))

	#   reading each frame of the video
    	firstFrame = None
    	firstFrame_new = None
    	oldFrame_new = None
    	t_start = None
    	t_final = None
    	cone_bottom_y_old = 500
    	min_area = 100
    	while True:
        	(grabbed, frame) = camera.read()
    		if grabbed == True:
        		frame = imutils.resize(frame, width = 900)
    			# warped image (Top-down view of the plate) using the computed transformation matrix
			warpedImage = cv2.warpPerspective(frame, transformationMatrix, (plate_size, plate_size))
               
			############## extracting t_start based on the first motion happend on the center of plate ##################
                	if t_start is None:
                		mask = np.zeros((warpedImage.shape[0], warpedImage.shape[1], 3), np.uint8)
				mask[400:600, 400:600] = warpedImage[400:600, 400:600]
                		gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
	        		gray = cv2.GaussianBlur(gray, (21, 21), 0)

				# if the first frame is None, initialize it
	      			if firstFrame is None:
	            			firstFrame = gray
		    			continue
                
                		# compute the absolute difference between the current frame and first frame to detect the motion
	        		frameDelta = cv2.absdiff(firstFrame, gray)
				thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	      	
				# dilate the thresholded image to fill in holes, then find contours on thresholded image
	      			thresh = cv2.dilate(thresh, None, iterations=2)
	      			(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	
   
	      			# loop over the contours
	      			for c in cnts:
	        	      		# if the contour is too small, ignore it
			     		if cv2.contourArea(c) < min_area:
		 	           		continue	
		      	      		# compute the bounding box for the contour, draw it on the frame,
		      	      		(x, y, w, h) = cv2.boundingRect(c)
		              		cv2.rectangle(warpedImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                	f_start = camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    					t_start = camera.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    					print f_start, t_start
			#################################################################                     
			if t_start is not None:
                		if t_final is None:   
                                	warpedImage_copy = warpedImage.copy()
					############## detection of the bottom of cone  #####################################
    					# Object detection using HSV red range color detector
    					obj_mask_1, obj_color_detected_1 = self.object_detection.Obj_color_detection(warpedImage, [0, 100, 100], [5, 255, 255])
    					obj_mask_2, obj_color_detected_2= self.object_detection.Obj_color_detection(warpedImage, [170, 100, 100], [180, 255, 255])
    					obj_mask = obj_mask_1 + obj_mask_2 
					cnts_c, hierarchy_c = self.object_detection.find_obj_contours(obj_mask)
                                	yc = []
    	   				if len(cnts_c) > 0:
						for i in cnts_c:
							x, y, w, h = cv2.boundingRect(i)
							cv2.rectangle(warpedImage_copy, (x, y), (x + w, y + h), (255, 0, 0), 1)
                                                	yc.append(y + h)

						cone_bottom = sorted(yc, reverse = True)
                                        	print cone_bottom
						cone_bottom_y = sorted(yc, reverse = True)[:1]
						cone_bottom_y_old = cone_bottom_y
                                        	print "cone bottom is :" + str(cone_bottom_y)
                                	else:
						cone_bottom_y = cone_bottom_y_old 

                                	#####################################################################################
					# Crop the image to detect motion box to have a better and accurate 
					#area to detect slump flow (the area below cone OR half of image, the warped image 1000x1000 pixel)
                			mask = np.zeros((warpedImage.shape[0], warpedImage.shape[1], 3), np.uint8)
					mask[500:900, 150:900] = warpedImage[500:900, 150:900]
                			gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
	        			gray = cv2.GaussianBlur(gray, (21, 21), 0)

					# if the first frame is None, initialize it
	      				if firstFrame_new is None:
	            				firstFrame_new = gray
		    				continue
               
                			# compute the absolute difference between the current frame and first frame
					frameDelta = cv2.absdiff(firstFrame_new, gray)
					thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]

					# dilate the thresholded image to fill in holes, then find contours on thresholded image
	      				thresh = cv2.dilate(thresh, None, iterations=2)
	      				(cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                			# compute the bounding box (w and h showing the new limited area that we are looking for concrete flow), draw it on the frame
                			x, y, w, h = cv2.boundingRect(max(cnts, key = cv2.contourArea))	
                        		cv2.rectangle(warpedImage_copy, (x, y), (x + w, y + h), (255, 255, 0), 2)
                        		cv2.drawContours(warpedImage_copy, [max(cnts, key = cv2.contourArea)], -1, (0,255,0), 1)

                        		# write the arpedImageframe ( the new detetction area)
        				video_out.write(warpedImage)
                        		#foreground.foreground_detection(warpedImage, x, y, w, h)

					############ Croping image to have a better area for concrete detection ############	
                        		new_mask = np.zeros((warpedImage.shape[0], warpedImage.shape[1], 3), np.uint8)
					new_mask[y - h: y + h, x: x + w] = warpedImage_copy[y - h: y + h, x: x + w]
                        	
					# bottom of cone method (the area bellow cone)
					new_mask_org = np.zeros((warpedImage.shape[0], warpedImage.shape[1], 3), np.uint8)
                        		new_mask_org[cone_bottom_y[0]: y + h, x: x + w] = warpedImage[cone_bottom_y[0]: y + h, x: x + w]
                                
					# OR half method (half of image, the warped image 1000x1000 pixel)
                        		#new_mask_org = np.zeros((warpedImage.shape[0], warpedImage.shape[1], 3), np.uint8)
                        		#new_mask_org[y: y + h, x: x + w] = warpedImage[y : y + h, x: x + w]

					# write the arpedImageframe
        				video_cropped.write(new_mask)
                        		#cv2.imshow('cropped', new_mask)
                        
					# ignore  small area:
					if w * h >= 20000:
                                		time = int(np.abs(camera.get(cv2.cv.CV_CAP_PROP_POS_MSEC) - t_start))

                                		################################### radius detection ###########################
                        			r, detected_area, d_ew, r_ns = self.radius.radius_detection(new_mask_org, cx, cy, cone_bottom_y[0] )
                                		#print r
                                		#cv2.imshow('detected', detected_area)
                                		# write the arpedImageframe
        		        		video_detected.write(detected_area)
                                		#foreground.foreground_detection(new_mask, x, y, w, h) 
 
						############################ create excel file ##################################
								
						for i in range(0, len(r), 20):
							self.data.append([time, r[i], d_ew, r_ns]) 

						############################     plot          ################################## 
			        		#fig, axes  = plt.subplots(ncols=2, nrows =2)
    						#ax0, ax1, ax2, ax3 = axes.ravel()

    						#img0 = ax0.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    						#ax0.set_title("Input image")
    						#ax0.axis("off")

    						#img1 = ax1.imshow(cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB))
    						#ax1.set_title("Warped image")
    						#ax1.axis("off")

    						#img2 = ax2.imshow(cv2.cvtColor(new_mask, cv2.COLOR_BGR2RGB))
    						#ax2.set_title("Foreground image")
    						#ax2.axis("off")

    						#img3 = ax3.imshow(cv2.cvtColor(detected_area, cv2.COLOR_BGR2RGB))
    						#ax3.set_title("Detected Area")
    						#ax3.axis("off")
	
    						#fig.tight_layout()
    						#plt.show()

                                	##################### extracting t_final #######################################
                			# compute the absolute difference between the current frame and old frame
					# if the old frame is None, initialize it
					#area to detect slump flow (the area below cone OR half of image, the warped image 1000x1000 pixel)
                			mask_n = np.zeros((warpedImage.shape[0], warpedImage.shape[1], 3), np.uint8)
					mask_n[100:800, 100:800]= warpedImage[100:800, 100:800]
                			warpedImage_gray = cv2.cvtColor(mask_n, cv2.COLOR_RGB2GRAY)
	        			warpedImage_gray = cv2.GaussianBlur(warpedImage_gray, (21, 21), 0)
	      				if oldFrame_new is None:
	            				oldFrame_new = warpedImage_gray
		    				continue
	        			frameDelta_final = cv2.absdiff(oldFrame_new, warpedImage_gray)
					thresh_final = cv2.threshold(frameDelta_final, 1, 255, cv2.THRESH_BINARY)[1]

					# dilate the thresholded image to fill in holes, then find contours on thresholded image
	      				thresh_final = cv2.dilate(thresh_final, None, iterations=2)
	      				(cnts_final, _) = cv2.findContours(thresh_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		                
                                	if len(cnts_final) == 0:
						f_final = camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    						t_final = camera.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    						print "t final is :" + str(t_final)
                                	else:
		      	      			# compute the bounding box for the contour, draw it on the frame,
		      	      			if cv2.contourArea(max(cnts_final, key = cv2.contourArea)) < 500:
							f_final = camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    							t_final = camera.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    							print "t final is :" + str(t_final)
						else:
							(x_final, y_final, w_final, h_final) = cv2.boundingRect(max(cnts_final, key = cv2.contourArea))
		              				cv2.rectangle(warpedImage, (x_final, y_final), (x_final + w_final, y_final + h_final), (0, 255, 255), 3)

					oldFrame_new = warpedImage_gray
                                        
                                	###################################################################################
                                                 

                	#cv2.imshow('warpedImage', warpedImage)

			if cv2.waitKey(1) & 0xFF == ord('q'):            		
				break
              
    		else:
        		break
    	# cleanup the camera and close any open windows
    	test_time = int(np.abs(t_final - t_start))
        cv2.destroyAllWindows()
    	camera.release()
    	video_out.release()
	video_cropped.release()
	video_detected.release()
    	#print data on a excel file
    	self.write_xls(self.filename_video, 'SCC_video', self.hdngs, self.data, self.heading_xf)
        return d_ew, test_time
 






