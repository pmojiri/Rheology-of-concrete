#!/usr/bin/python

##############################################################################################
# File name : perspective_cone.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - Perspective and Perspective_cone classes
#              - Cone detection using the marked area (red color tape) 
#  	       - four corners are used to compute the cone trasformation matrix             
#	       - Inputs are an image and new image size 
# 	       - outputs are the cone transformation matrix, warped image, top/bottom point of cone
#
###############################################################################################
# import the necessary packages
import cv2
import numpy as np
import imutils
from skimage.filters import threshold_otsu, threshold_adaptive
import sys
sys.path.append("/home/viki/Desktop/Slump_test/packages")

from Image_functions import image_functions
from Object_detector import object_detector
import yaml
import json
################################################################################
#     change the perspective look of an image
class Perspective(object):
    def __init__(self, source, source_color, new_image_size):
        self.source = source
        self.source_color = source_color
        self.corners = []
        self.corner = []
        self.refined_corners = []
        self.image_size = new_image_size  #new image sieze in pixel
	self.object_detection = object_detector()

#     find the centroid of the points of intersections found
    def calculateCentroid(self, corners): 
        M = cv2.moments(corners)
        print M
        if M['m00'] == 0:
           M['m00'] = 0.01
        corners_centroidx = int(M['m10']/M['m00'])
        corners_centroidy = int(M['m01']/M['m00'])
        print "centroids", corners_centroidx, corners_centroidy

        return [corners_centroidx, corners_centroidy]      
    
#     find the Top right, Top left, Bottom right and Bottom left points
    def calculateTRTLBRBL(self, cx, cy):
        topoints = []
        bottompoints = []
        cx = cx
        cy = cy
        print "coordinates   " + str(self.corners)
        for i in self.corners:
            print i, type(i)
            print cy
            if i[1] < cy:
                topoints.append(i)
            else: 
                bottompoints.append(i)
        print "top points" + str(topoints)
        print "bottom points" + str(bottompoints)
        top_left = min(topoints)
        top_right = max(topoints)
        bottom_right = max(bottompoints)
        bottom_left = min(bottompoints)
        
        corners.append(top_left)
        corners.append(top_right)
        corners.append(bottom_right)
        corners.append(bottom_left)
        print corners

        return corners
        
#     get the destinations and edges
    def handle(self):
        img = self.source
        self.shape = img.shape
        width = self.shape[1]
        height = self.shape[0]           

    	# Object detection using HSV red range color detector
    	obj_mask_1, obj_color_detected_1 = self.object_detection.Obj_color_detection(img, [0, 100, 100], [5, 255, 255])
    	obj_mask_2, obj_color_detected_2= self.object_detection.Obj_color_detection(img, [170, 100, 100], [180, 255, 255])
    	obj_mask = obj_mask_1 + obj_mask_2 
    	obj_color_detected = obj_color_detected_1 + obj_color_detected_2
        #kernel = np.ones((6, 6), np.uint8)
        #obj_mask = cv2.dilate(obj_mask, kernel, None)
    	cv2.imwrite("images/result_images/Masked.jpg", obj_mask)

        #cv2.imshow("Masked", obj_mask)
        #cv2.waitKey(0)

        return obj_mask
        
#   find the corners using contours method
    def contourmethod(self, obj_mask):
    	contours_c, hierarchy_c = self.object_detection.find_obj_contours(obj_mask)
    	c = sorted(contours_c, key = cv2.contourArea, reverse = True)[:3]
    	print c
    	print len(c)
    	cv2.drawContours(self.source, c[0], -1, (0,255,0), 3)
    	cv2.drawContours(self.source, c[1], -1, (0,255,255), 3)
    	cv2.drawContours(self.source, c[2], -1, (255,255,0), 3)

	topmost_c0 = tuple(c[0][c[0][:,:,1].argmin()][0])
	topmost_c1 = tuple(c[1][c[0][:,:,1].argmin()][0])
	topmost_c2 = tuple(c[2][c[0][:,:,1].argmin()][0])
     
        if (topmost_c0[1] < topmost_c1[1]) and (topmost_c0[1] < topmost_c2[1]):
           start_pixel = 1000
           self.destination = np.float32([[start_pixel, start_pixel], [start_pixel + 120, start_pixel],  [start_pixel + 170, start_pixel + 290], [start_pixel - 40, start_pixel + 290]])
           if topmost_c1[0] < topmost_c2[0]:
              top_cnt = c[0]
              bottomleft_cnt = c[1]
              bottomright_cnt = c[2]
           else:
              top_cnt = c[0]
              bottomleft_cnt = c[2]
              bottomright_cnt = c[1]

    	   #top_cnt_approx = cv2.approxPolyDP(top_cnt, 0.01 * cv2.arcLength(top_cnt, True), True)
    	   #cv2.drawContours(self.source, top_cnt_approx, -1, (255,255,255), 6)
    	   #top_cnt_sorted = sorted(top_cnt_approx, key=lambda x: x[0][1])
           #if top_cnt_sorted[0][0][0] < top_cnt_sorted[1][0][0]:
           #   self.corners.append([top_cnt_sorted[0][0][0], top_cnt_sorted[0][0][1]])
           #   self.corners.append([top_cnt_sorted[1][0][0], top_cnt_sorted[1][0][1]])
           #else:
           #   self.corners.append([top_cnt_sorted[1][0][0], top_cnt_sorted[1][0][1]])
           #   self.corners.append([top_cnt_sorted[0][0][0], top_cnt_sorted[0][0][1]])

    	   #cv2.circle(self.source, (top_cnt_sorted[0][0][0], top_cnt_sorted[0][0][1]), 10, (255, 255, 0), 2)
    	   #cv2.circle(self.source, (top_cnt_sorted[1][0][0], top_cnt_sorted[1][0][1]), 10, (255, 255, 0), 2)

    	   #leftmost = tuple(c[0][c[0][:,:,0].argmin()][0])
    	   #rightmost = tuple(c[0][c[0][:,:,0].argmax()][0])
    	   #topmost = tuple(c[0][c[0][:,:,1].argmin()][0])
    	   #bottommost = tuple(c[0][c[0][:,:,1].argmax()][0])
    
    	   x_t, y_t, w_t, h_t = cv2.boundingRect(top_cnt)
           cv2.rectangle(self.source, (x_t, y_t), (x_t + w_t, y_t + h_t), (255, 0, 0), 1)
           self.corners.append([x_t, y_t])
           self.corners.append([x_t + w_t, y_t])
    	   cv2.circle(self.source, (x_t, y_t), 10, (255, 255, 0), 2)
    	   cv2.circle(self.source, (x_t + w_t, y_t), 10, (255, 255, 0), 2)

      	   x_br, y_br, w_br, h_br = cv2.boundingRect(bottomright_cnt)
      	   cv2.rectangle(self.source, (x_br, y_br), (x_br + w_br, y_br + h_br), (255, 0, 0), 1)
      	   center_point_bottomright = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), [(x_br, y_br), (x_br + w_br, y_br + h_br)])

           corners_bottomright = (center_point_bottomright[0] - (w_br/2) , center_point_bottomright[1]) 
           self.corners.append([corners_bottomright[0], corners_bottomright[1]])
      	   cv2.circle(self.source, corners_bottomright, 10, (0, 255, 0), 2)


      	   x_bl, y_bl, w_bl, h_bl = cv2.boundingRect(bottomleft_cnt)
      	   cv2.rectangle(self.source, (x_bl, y_bl), (x_bl + w_bl, y_bl + h_bl), (255, 0, 0), 1)
       	   center_point_bottomleft = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), [(x_bl, y_bl), (x_bl + w_bl, y_bl + h_bl)])
       	   corners_bottomleft = (center_point_bottomleft[0] + (w_bl/2) , center_point_bottomleft[1])
           self.corners.append([corners_bottomleft[0], corners_bottomleft[1]])
       	   cv2.circle(self.source, corners_bottomleft, 10, (0, 255, 0), 2)
	


        if (topmost_c0[1] > topmost_c1[1]) and (topmost_c0[1] > topmost_c2[1]):
	   start_pixel = 1000
           self.destination = np.float32([[start_pixel, start_pixel], [start_pixel + 200, start_pixel],  [start_pixel + 140, start_pixel + 290], [start_pixel + 40, start_pixel + 290]])
           if topmost_c1[0] < topmost_c2[0]:
              bottom_cnt = c[0]
              topleft_cnt = c[1]
              topright_cnt = c[2]
           else:
              bottom_cnt = c[0]
              topleft_cnt = c[2]
              topright_cnt = c[1]

      	   x_bl, y_bl, w_bl, h_bl = cv2.boundingRect(topleft_cnt)
      	   cv2.rectangle(self.source, (x_bl, y_bl), (x_bl + w_bl, y_bl + h_bl), (255, 0, 0), 3)
       	   center_point_topleft = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), [(x_bl, y_bl), (x_bl + w_bl, y_bl + h_bl)])
       	   corners_topleft = (center_point_topleft[0] + (w_bl/2) , center_point_topleft[1])
           self.corners.append([corners_topleft[0], corners_topleft[1]])
       	   cv2.circle(self.source, corners_topleft, 10, (0, 255, 0), 2)


      	   x_br, y_br, w_br, h_br = cv2.boundingRect(topright_cnt)
      	   cv2.rectangle(self.source, (x_br, y_br), (x_br + w_br, y_br + h_br), (255, 0, 0), 3)
      	   center_point_topright = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), [(x_br, y_br), (x_br + w_br, y_br + h_br)])

           corners_topright = (center_point_topright[0] - (w_br/2) , center_point_topright[1]) 
           self.corners.append([corners_topright[0], corners_topright[1]])
      	   cv2.circle(self.source, corners_topright, 10, (0, 255, 0), 2)


    	   #bottom_cnt_approx = cv2.approxPolyDP(bottom_cnt, 0.01 * cv2.arcLength(bottom_cnt, True), True)
    	   #cv2.drawContours(self.source, bottom_cnt_approx, -1, (255,255,255), 6)
    	   #bottom_cnt_sorted = sorted(bottom_cnt_approx, key=lambda x: x[0][1], reverse = True)
           #if bottom_cnt_sorted[0][0][0] > bottom_cnt_sorted[1][0][0]:
           #   self.corners.append([bottom_cnt_sorted[0][0][0], bottom_cnt_sorted[0][0][1]])
           #   self.corners.append([bottom_cnt_sorted[1][0][0], bottom_cnt_sorted[1][0][1]])
           #else:
           #   self.corners.append([bottom_cnt_sorted[1][0][0], bottom_cnt_sorted[1][0][1]])
           #   self.corners.append([bottom_cnt_sorted[0][0][0], bottom_cnt_sorted[0][0][1]])

    	   #cv2.circle(self.source, (bottom_cnt_sorted[0][0][0], bottom_cnt_sorted[0][0][1]), 10, (255, 255, 0), 2)
    	   #cv2.circle(self.source, (bottom_cnt_sorted[1][0][0], bottom_cnt_sorted[1][0][1]), 10, (255, 255, 0), 2)

    	   x_t, y_t, w_t, h_t = cv2.boundingRect(bottom_cnt)
           cv2.rectangle(self.source, (x_t, y_t), (x_t + w_t, y_t + h_t), (255, 0, 0), 1)
           self.corners.append([x_t, y_t + (h_t/2)])
           self.corners.append([x_t + w_t, y_t + (h_t/2)])
    	   cv2.circle(self.source, (x_t, y_t + (h_t/2)), 10, (255, 255, 0), 2)
    	   cv2.circle(self.source, (x_t + w_t, y_t + (h_t/2)), 10, (255, 255, 0), 2)

        
    	#cv2.imwrite("images/result_images/Masked_contours.jpg", self.source)
        #cv2.imshow("Masked_contours", self.source)
        cv2.waitKey(0)

        print self.corners
        return self.corners

    
#     transform the points to the destination and return warped image and transformationMatrix
    def transform(self, corners, source_2):
        corners = np.float32((corners[0], corners[1], corners[2], corners[3]))
#         print "transform", corners[0][0], corners[1][0], corners[2][0], corners[3][0]
#         corners = np.float32(corners)
        transformationMatrix = cv2.getPerspectiveTransform(corners, self.destination)
        minVal = np.min(self.destination[np.nonzero(self.destination)])
        print "minVal", minVal, "width", self.shape[0]
        maxVal = np.max(self.destination[np.nonzero(self.destination)])
        print "maxVal", maxVal, "height", self.shape[1]
        warpedImage = cv2.warpPerspective(source_2, transformationMatrix, (self.image_size, self.image_size))
        return warpedImage, transformationMatrix
        
#   improve the image by sharpening it
    def showsharpen(self, warpedImage):
        #cv2.imshow("image", warpedImage)
        #cv2.waitKey(0)
        # gray = cv2.cvtColor(warpedImage, cv2.cv.CV_BGR2GRAY)
        blur = cv2.GaussianBlur(warpedImage, (5, 5), 2)
        alpha = 1.5
        beta = 1 - alpha #1 - alpha
        gamma = 0
        sharpened = cv2.addWeighted(warpedImage, alpha, blur, beta, gamma)
        cv2.imwrite("images/result_images/warped_cone.jpg", sharpened)
        #cv2.imshow("sharpened", sharpened)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()

###################################################################################################################
#     change the perspective look of an image
class Perspective_cone(object):
    def __init__(self, source_color, new_image_size):
        self.source_color = source_color
        self.source_color = imutils.resize(self.source_color, width = 900)
        self.new_image_size = new_image_size  #new image size in pixcel
	self.source = source_color
	self.source_2 = source_color
	self.source = imutils.resize(self.source, width = 900)
	self.source_2 = imutils.resize(self.source_2, width = 900)
        #cv2.imshow("Original image", self.source_color)
	#cv2.waitKey(0)

    def cone_detection(self):
	#self.source = cv2.GaussianBlur(self.source, (3, 3), 0)
	persp = Perspective(self.source, self.source_color, self.new_image_size)
	obj_mask = persp.handle()
	corners = persp.contourmethod(obj_mask)
	print "corners are:" + str(corners)
	#print len(corners)
	for i in xrange(0, len(corners)):
      		print corners[i]
      		cv2.circle(self.source_2, (corners[i][0], corners[i][1]), 6, (255, 105, 0), 2)
	cv2.imwrite("images/result_images/cone_corners.jpg", self.source_2)
	#cv2.imshow("cone_corners", self.source_2)
	#cv2.waitKey(0)

	warpedImage, transformationMatrix = persp.transform(corners, self.source_2)
	persp.showsharpen(warpedImage)

	data = {"homography_matrix": transformationMatrix.tolist()}
	with open('files/transformationMatrix_cone.yaml', "w") as f:
		yaml.dump(data, f)
	with open('files/transformationMatrix_cone.json', "w") as f1:
		json.dump(data, f1)


	# computing top and bottom image point of cone
	p1 = np.array([[corners[0][0] + ((corners[1][0] - corners[1][0])/2.0)], [corners[0][1]], [1]])
	p2 = np.array([[corners[3][0] + ((corners[2][0] - corners[3][0])/2.0)], [corners[3][1]], [1]])

        x_top = corners[0][1]
        x_bottom = corners[3][1]
        bottom_point = p2

	# computing top and bottom real point of cone (using cone transformation matrix)
	P1 = np.dot(transformationMatrix, p1)
	P2 = np.dot(transformationMatrix, p2)

	P_top = (P1[0, 0] / P1[2, 0], P1[1, 0] / P1[2, 0])
	P_bottom = (P2[0, 0] / P2[2, 0], P2[1, 0] / P2[2, 0])

	# computing cone height
        cone_height = np.sqrt(np.power((P_bottom[0] - P_top[0]),2) + np.power((P_bottom[1] - P_top[1]),2))
	print "cone_height is: " + str(cone_height)
        
        return warpedImage, transformationMatrix, cone_height, x_top, x_bottom, bottom_point

