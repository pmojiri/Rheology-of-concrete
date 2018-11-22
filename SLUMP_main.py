#!/usr/bin/python

######################################################################################
# File name : SLUMP_main.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: The main code to run SLUMP test
#              inputs: - an image directory or a video directory
#		       - plate size in milimeter
#		       - new image size in pixel (default is 3000)	
#              output: - Excel file showing the requested data
#
# USAGE: - python SCC_main.py
# 	 - python SCC_main.py --image images/example_01.jpg --platesize 1000 --imagesize 3000
# 	 - python SCC_main.py --video videos/example_01.mp4 --platesize 1000 --imagesize 3000
#
###########################################################################################
#
# import the necessary packages
import cv2
import numpy as np
import imutils
import argparse
import sys
import datetime
import time
import yaml
import json
import xlwt
from matplotlib import pyplot as plt
from imutils.video import VideoStream

sys.path.append("/home/viki/Desktop/Slump_test/packages")
from perspective import Perspective
from perspective import Perspective_plate

from Image_functions import image_functions
from Video_functions import video_functions
from Foreground_detector import foreground_detector
from Radius_detector import radius_detector

from perspective_cone import Perspective
from perspective_cone import Perspective_cone
from Height_detector import height_detector
from Object_detector import object_detector
#
################################################################################
#
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-ps", "--platesize", type=int, default = 1000, help="plate size in milimeter")
ap.add_argument("-is", "--imagesize", type=int, default = 3000, help="new image size in pixel")
args = vars(ap.parse_args())
#
################################################################################
#
# initialization
image = image_functions()
camera = video_functions()
foreground = foreground_detector()
radius = radius_detector()
height = height_detector()
object_detection = object_detector()

filename_image = 'excel_files/SLUMP_test5_data_image.xls'
filename_video = 'excel_files/SLUMP_test5_data_video.xls'
ezxf = xlwt.easyxf
hdngs = ['  Time (s)   ', '   edge distance from center point   ', '     Diameter_ew (mm)   ', '     radius_ns (mm)     ', '     Height (mm)   ']
heading_xf = ezxf('font: bold on; align: wrap on, vert centre, horiz center')
data = []
#
################################################################################
#
# write data on Excel file
def write_xls(file_name, sheet_name, headings, data, heading_xf):
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
#
################################################################################
#
# if the video argument is None, then we are reading image file (The image file should be the final image detecting four corners of the plate, complete cone, and the slipped concrete:
if args.get("video", None) is None:
    # reading the inputs:
    image = image.read_image(args["image"])
    plate_size = args["platesize"]
    new_image_size = args["imagesize"]

    # computing trasformation matrix calling houghlineP method (plate perspective) and warped image (Top-down view of the plate)
    persp_plate = Perspective_plate(image, plate_size)
    warpedImage_plate, transformationMatrix_plate = persp_plate.houghlinePmethod()

    # extracting the foreground from background to have a better concrete detetction (manually defined a box)
    warpedImage_cropped = foreground.foreground_detection(warpedImage_plate, (plate_size/2 - 350), (plate_size/2 - 350), (plate_size/2 + 200), (plate_size/2 + 200))

    # concrete detection to extract the radius
    cx = warpedImage_plate.shape[1]/2
    cy = warpedImage_plate.shape[0]/2
    r, detected_area, d_ew, r_ns = radius.radius_detection(warpedImage_cropped, cx, cy, cy)
    cv2.imshow('radius_detected', detected_area)
    cv2.waitKey(0)

    # save the detected area
    cv2.imwrite("images/result_images/slump_radius_detected.jpg", detected_area)

    # showing the results
    fig, axes  = plt.subplots(ncols=2, nrows =2)
    ax0, ax1, ax2, ax3 = axes.ravel()

    img0 = ax0.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax0.set_title("Input image")
    ax0.axis("off")

    img1 = ax1.imshow(cv2.cvtColor(warpedImage_plate, cv2.COLOR_BGR2RGB))
    ax1.set_title("Warped image")
    ax1.axis("off")

    img2 = ax2.imshow(cv2.cvtColor(warpedImage_cropped, cv2.COLOR_BGR2RGB))
    ax2.set_title("Foreground image")
    ax2.axis("off")

    img3 = ax3.imshow(cv2.cvtColor(detected_area, cv2.COLOR_BGR2RGB))
    ax3.set_title("Detected Area")
    ax3.axis("off")

    fig.tight_layout()
    plt.show()

    # computing trasformation matrix of cone (cone perspective)
    persp_cone = Perspective_cone(image, new_image_size)
    warpedImage_cone, transformationMatrix_cone, cone_height, x_top, x_bottom, bottom_point = persp_cone.cone_detection()
    frame = imutils.resize(image, width = 900)

    # concrete detection to extract the height
    h, height_detected = height.height_detection(frame, x_top, x_bottom)
    cv2.imshow('height_detected', height_detected)
    cv2.waitKey(0)

    # save the detected area and write the data on a excel file
    cv2.imwrite("images/result_images/slump_height_detected.jpg", detected_area)
    for i in xrange(0, len(r)):
	data.append([0, r[i], d_ew, r_ns, int(np.abs(300 - h))])
    write_xls(filename_image, 'SLUMP_image', hdngs, data, heading_xf)

    # showing the results
    fig, axes  = plt.subplots(ncols=2, nrows =2)
    ax0, ax1, ax2, ax3 = axes.ravel()

    img0 = ax0.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax0.set_title("Input image")
    ax0.axis("off")

    img1 = ax1.imshow(cv2.cvtColor(warpedImage_cone, cv2.COLOR_BGR2RGB))
    ax1.set_title("Warped image")
    ax1.axis("off")

    img2 = ax2.imshow(cv2.cvtColor(height_detected, cv2.COLOR_BGR2RGB))
    ax2.set_title("Height Detected area")
    ax2.axis("off")

    img3 = ax3.imshow(cv2.cvtColor(height_detected, cv2.COLOR_BGR2RGB))
    ax3.set_title("Height Detected area")
    ax3.axis("off")

    fig.tight_layout()
    plt.show()


# otherwise, we are reading from a video
else:
    # reading the inputs:  
    camera = camera.read_video(args["video"])
    plate_size = args["platesize"]
    new_image_size = args["imagesize"]

    # Grab first frame of the video 
    (first_grabbed, first_frame) = camera.read()
    pos_frame = camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    pos_time = camera.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    print pos_frame, pos_time
    cv2.imwrite("images/result_images/SLUMP_Sfirst_frame.jpg", first_frame)

    # computing trasformation matrix calling houghlineP method (plate perspective) and warped image (Top-down view of the plate)
    persp_plate = Perspective_plate(first_frame, plate_size)
    warpedImage_plate, transformationMatrix_plate = persp_plate.houghlinePmethod()

    # computing trasformation matrix of cone (cone perspective)
    persp_cone = Perspective_cone(first_frame, new_image_size)
    warpedImage_cone, transformationMatrix_cone, cone_height, x_top, x_bottom, bottom_point = persp_cone.cone_detection()

    p1 = bottom_point
    P1 = np.dot(transformationMatrix_plate, p1)
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
    video_out = cv2.VideoWriter("videos/SLUMP_test5_warped_plate.avi", fourcc, 30, (plate_size, plate_size))
    video_cropped = cv2.VideoWriter("videos/SLUMP_test5_cropped_plate.avi", fourcc, 30, (plate_size, plate_size))
    video_detected = cv2.VideoWriter("videos/SLUMP_test5_radius_detected.avi", fourcc, 30, (plate_size, plate_size))

    first_frame = imutils.resize(first_frame, width = 900)
    video_out_cone = cv2.VideoWriter("videos/SLUMP_test5_warped_cone.avi", fourcc, 30, (first_frame.shape[1], first_frame.shape[0]))
    video_height_detected = cv2.VideoWriter("videos/SLUMP_test5_height_detected.avi", fourcc, 30, (new_image_size, new_image_size))

#   reading each frame of the video
    firstFrame = None
    firstFrame_new = None
    oldFrame_new = None
    t_start = None
    t_final = None
    min_area = 100
    while True:
        (grabbed, frame) = camera.read()
    	if grabbed == True:
        	frame = imutils.resize(frame, width = 900)
    		# warped image (Top-down view of the plate) using the computed transformation matrix
		warpedImage_plate = cv2.warpPerspective(frame, transformationMatrix_plate, (plate_size, plate_size))
    		# warped image using the computed transformation matrix
		warpedImage_cone = cv2.warpPerspective(frame, transformationMatrix_cone, (3000, 3000))
		cv2.imwrite("images/result_images/warped_cone_final.jpg", warpedImage_cone)

		############## extracting t_start based on the first motion happend on the center of plate ##################
                if t_start is None:
                	mask = np.zeros((warpedImage_plate.shape[0], warpedImage_plate.shape[1], 3), np.uint8)
			mask[400:600, 400:600] = warpedImage_plate[400:600, 400:600]
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
		              	cv2.rectangle(warpedImage_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                f_start = camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    				t_start = camera.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    				print f_start, t_start
		############################################################################################### 
		if t_start is not None:
                	if t_final is None: 
                                #warpedImage_copy = warpedImage_plate.copy()
				# Crop the image to detect motion box to have a better and accurate 
				#area to detect slump flow (the area below cone OR half of image, the warped image 1000x1000 pixel)
                		mask = np.zeros((warpedImage_plate.shape[0], warpedImage_plate.shape[1], 3), np.uint8)
				mask[500:900, 150:900] = warpedImage_plate[500:900, 150:900]
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
	
                		# compute the bounding box (w and h showing the new limited area that we are looking for concrete flow), draw it on the frames
                		x, y, w, h = cv2.boundingRect(max(cnts, key = cv2.contourArea))	
				warpedImage_plate_copy = warpedImage_plate.copy()
                        	cv2.rectangle(warpedImage_plate_copy, (x, y), (x + w, y + h), (255, 255, 0), 2)

                        	cv2.drawContours(warpedImage_plate_copy, [max(cnts, key = cv2.contourArea)], -1, (0,255,0), 1)

                        	# write the arpedImageframe
        			video_out.write(warpedImage_plate)
                        	#foreground.foreground_detection(warpedImage_plate, x, y, w, h)

				############ Croping image to have a better area for concrete detection ############
                        	new_mask = np.zeros((warpedImage_plate.shape[0], warpedImage_plate.shape[1], 3), np.uint8)
				new_mask[y - h: y + h, x: x + w] = warpedImage_plate_copy[y - h: y + h, x: x + w]

				# bottom of cone method (the area bellow cone)
                        	new_mask_org = np.zeros((warpedImage_plate.shape[0], warpedImage_plate.shape[1], 3), np.uint8)
                        	new_mask_org[cone_bottom[1]: y + h, x: x + w] = warpedImage_plate[cone_bottom[1]: y + h, x: x + w]

				# write the arpedImageframe
        			video_cropped.write(new_mask)
                        	cv2.imshow('cropped', new_mask)
                        
				if w * h >= 20000:

                                	time = int(np.abs(camera.get(cv2.cv.CV_CAP_PROP_POS_MSEC) - t_start))
                                	################################### radius detection ###########################

                        		r, detected_area, d_ew, r_ns = radius.radius_detection(new_mask_org, cx, cy, cone_bottom[1])
                                	#print r
                                	cv2.imshow('radius_detected', detected_area)
                                	# write the arpedImageframe
        		        	video_detected.write(detected_area)
                                	#foreground.foreground_detection(new_mask, x, y, w, h)

			        	############################ height detection ##################################
                                
                        		#print x_top, x_bottom
                        		#cv2.imshow('frame', frame)
                        		#cv2.waitKey(0)
					h, height_detected = height.height_detection(frame, x_top, x_bottom)
                        		#print h
                        		cv2.imshow('height_detected', height_detected)
                                	# write the arpedImageframe
        		        	video_height_detected.write(height_detected)
	
					############################ create excel file ##################################

					slump_height = int(np.abs(300 - h))
                                	if slump_height <= 20.0 :
                                		slump_height = 0.0

    					for i in xrange(0, len(r)):
						data.append([time, r[i], d_ew, r_ns, slump_height])

                                ##################### extracting t_final #######################################
                		# compute the absolute difference between the current frame and old frame
				# if the old frame is None, initialize it
                		mask_n = np.zeros((warpedImage_plate.shape[0], warpedImage_plate.shape[1], 3), np.uint8)
				mask_n[100:800, 100:800]= warpedImage_plate[100:800, 100:800]
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
		      	      		if cv2.contourArea(max(cnts_final, key = cv2.contourArea)) < min_area:
						f_final = camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    						t_final = camera.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    						print "t final is :" + str(t_final)
					else:
						(x_final, y_final, w_final, h_final) = cv2.boundingRect(max(cnts_final, key = cv2.contourArea))
		              			cv2.rectangle(warpedImage_plate, (x_final, y_final), (x_final + w_final, y_final + h_final), (0, 255, 255), 3)

				oldFrame_new = warpedImage_gray
                                ###################################################################################
                                

                cv2.imshow('warpedImage_plate', warpedImage_plate)

		if cv2.waitKey(1) & 0xFF == ord('q'):            		
			break
              
    	else:
        	break
    # cleanup the camera and close any open windows
    cv2.destroyAllWindows()
    camera.release()
    video_out.release()
    #print data on a excel file
    write_xls(filename_video, 'SLUMP_video', hdngs, data, heading_xf)
 






