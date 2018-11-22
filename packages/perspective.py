#!/usr/bin/python

####################################################################################
# File name : perspective.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - Perspective and Perspective_plate classes
#              - Three methods are prestected to detect four corners of plate: 
#			- HOUGH LINE TRANSFORM METHOD
#			- Probabilistic HOUGH LINE TRANSFORM METHOD
#			- CONTOUR METHOD 
#  	       - four corners are used to compute the plate trasformation matrix             
#	       - Inputs are an image and plate size 
# 	       - outputs are the top-down view image and the plate transformation matrix	
#
####################################################################################
# import the necessary packages
import cv2
import numpy as np
import imutils
from skimage.filters import threshold_otsu, threshold_adaptive
import sys
sys.path.append("/home/viki/Desktop/Slump_test/packages")

from coordinates import Coordinates
from intersections import Intersections
################################################################################
#     change the perspective look of an image
class Perspective(object):
    def __init__(self, source, source_color, platesize, platecolor_min, platecolor_max):
        self.source = source
        self.source_color = source_color
        self.corners = []
        self.refined_corners = []
	self.plate_color_min = platecolor_min
	self.plate_color_max = platecolor_max
        self.plate_size = platesize  #plate squre size in centimeter
        
#     get the destinations and edges showing the plate lines
    def handle(self):
        img = self.source
        self.shape = img.shape
        width = self.shape[1]
        height = self.shape[0]           
        self.destination = np.float32([[0,0], [self.plate_size, 0],  [self.plate_size, self.plate_size], [0, self.plate_size]])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # black color detection (tape color around the plate)
    	COLOR_MIN = np.array(self.plate_color_min,np.uint8) 
    	COLOR_MAX = np.array(self.plate_color_max,np.uint8)
    	img_mask = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
        kernel = np.ones((3, 3), np.uint8)
        img_mask = cv2.dilate(img_mask, kernel, None)
        #img = cv2.bitwise_and(img, img, mask = img_mask)
        #cv2.imshow("black color detection", img_mask)
        #cv2.waitKey(0)

	# Canny edge detection using automatice parameter (median values)
	##edges = cv2.adaptiveThreshold(cv2.cvtColor(self.source,cv2.COLOR_BGR2GRAY),255,1,1,11,2)
        sigma = 0.33
        v = np.median(img)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        #lower = threshold_otsu(img)
        #upper = threshold_otsu(img) * 0.5
        edges = cv2.Canny(img, lower, upper, apertureSize = 3)
        edges = cv2.convertScaleAbs(edges)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        kernel = np.ones((3, 3), np.uint8)
        #edges = cv2.dilate(edges, kernel, None)
        #edges = cv2.erode(edges, kernel, None)

	# combine black color detection and Canny edge detetcion
        edges = cv2.bitwise_and(img_mask, img_mask, mask = edges)
        cv2.imwrite("images/result_images/Edges.jpg", edges)
        #cv2.imshow("edges", edges)
        #cv2.waitKey(0)

        # ignore small areas and compute the edges again (combination of new image and canny edge detection)
	contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.source_color, contours, -1, (0,255,255), 1)
        #cv2.imshow("contours", self.source_color)
        #cv2.waitKey(0)
        mask = np.ones(self.source_color.shape[:2], dtype="uint8") * 255
        new_edges = []
        for c in contours:
           if cv2.contourArea(c) > 250:
              cv2.drawContours(mask, [c], -1, 0, -1)              
        image = cv2.bitwise_and(self.source_color, self.source_color, mask=mask)
        #cv2.imshow("Mask", mask)
        #cv2.waitKey(0)
	#cv2.imshow("After", image)
        #cv2.waitKey(0)

        sigma = 1
        v = np.median(mask)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(mask, lower, upper, apertureSize = 3)
        edges = cv2.convertScaleAbs(edges)
        #cv2.imshow("new edges", edges)
        #cv2.waitKey(0)

        return edges
        
#   find the hough lines and the intersection points
    def houghlinemethod(self, edges):
        lines = cv2.HoughLines(edges, 0.7, np.pi/500, 130)
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(self.source_color,(x1,y1),(x2,y2),(0,255,255),1)

        # Initial Intersection class
	inter = Intersections()
        for i in range(0, lines.shape[1]):
            for j in range(i + 1, lines.shape[1]):
                line1 = lines[0][i]
                line2 = lines[0][j]
                # check the lines as a pair to compute their intersections
                if inter.acceptLinePair(line1, line2, np.pi/32):
                        intersection = inter.computeintersect(line1, line2)
                        #print intersection
                        if (intersection[0] > 0 and intersection[1] > 0):
                             inter.append(intersection[0], intersection[1])
                             cv2.circle(self.source_color, (intersection[0], intersection[1]), 6, (0, 255, 255), 2)


        # combination of the extracted intersection with harris corner detetction
        gray = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)
        '''harris corner detector and draw red circle around them'''
        self.features = cv2.goodFeaturesToTrack(gray, 1000, 0.001, 5, None, None, 2, useHarrisDetector = True, k = 0.00009)
        if len(self.features.shape) == 3.0:
           assert(self.features.shape[1:] ==(1,2))
           self.features.shape = (self.features.shape[0], 2)

        for x, y in self.features:
           self.corners += [(x , y)]
           inter.append_features(x , y)
           cv2.circle(self.source_color, (x, y), 8, (255, 255, 255))

        cv2.imwrite("images/result_images/Lines_intersection.jpg", self.source_color)
        #cv2.imshow("lines and intersections", self.source_color)
        #cv2.waitKey(0)

       	# subpixel accuracy
	'''define the criteria to stop and refine the corners'''
    	criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.03)
    	#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    	cv2.cornerSubPix(gray, self.features, (5, 5), (-1, -1), criteria)
    
    	for x, y in self.features:
       		self.refined_corners += [(x , y)]
                inter.append_refined_features(x , y)
       		cv2.circle(self.source_color, (x, y), 4, (0, 255, 0))

    	'''Now draw them'''
    	res = np.hstack ((self.corners, self.refined_corners))
    	res = np.int0(res)
    	self.source_color[res[:,1], res[:, 0]] = [0, 0, 255]
    	self.source_color[res[:,3], res[:, 2]] = [0, 255, 0]
    	#cv2.imwrite(str(self.IMAGE_NAME) + '_subpixel.png', self.im_new)

    	#cv2.imshow("corners", self.source_color) 
        #cv2.waitKey(0)

        return inter


#   find the probabilistic hough lines and the intersection points
    def houghlinePmethod(self, edges):
        minLineLength = int(self.source_color.shape[0] / 5.0)
        maxLineGap = int(self.source_color.shape[0] / 2.0)
        linesP = cv2.HoughLinesP(edges, 1, np.pi/180, 95, np.array([]), minLineLength, maxLineGap)
        print "linesP are:   " + str(linesP)
        for x1,y1,x2,y2 in linesP[0]:
             #cv2.circle(self.source_color, (x1, y1), 6, (0, 255, 255), 1)
             #cv2.circle(self.source_color, (x2, y2), 6, (255, 0, 255), 1)
             cv2.line(self.source_color,(x1,y1),(x2,y2),(0, 0, 255))

        #cv2.imshow("lines", self.source_color)
        #cv2.waitKey(0)

        # Initial Intersection class
        inter = Intersections()
        d = []
        for i in range(0, linesP.shape[1]):
            for j in range(i + 1, linesP.shape[1]):
                line1 = linesP[0][i]
                line2 = linesP[0][j] 
                # check the lines as a pair to compute their intersections               
                if inter.acceptLinesPPair(line1, line2, np.pi/20.0):              
                        intersection = inter.computeintersectP(line1, line2)
                        print intersection
                        if ((intersection[0] > 0 and intersection[0] < self.source_color.shape[1]) and (intersection[1] > 0 and intersection[1] <  self.source_color.shape[0])):
                        #if ((intersection[0] > 0) and (intersection[1] > 0)):
                             inter.append(intersection[0], intersection[1])
                             cv2.circle(self.source_color, (intersection[0], intersection[1]), 10, (0, 255, 0))
                             #cv2.line(self.source_color,(line1[0],line1[1]),(line1[2],line1[3]),(255, 255, 0))
                             #cv2.line(self.source_color,(line2[0],line2[1]),(line2[2],line2[3]),(255, 255, 0))

        cv2.imwrite("images/result_images/Lines_intersection.jpg", self.source_color)
        #cv2.imshow("lines and intersections", self.source_color)
        #cv2.waitKey(0)

        # combination of the extracted intersection with harris corner detetction
	gray = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)
        '''harris corner detector and draw red circle around them'''
        self.features = cv2.goodFeaturesToTrack(gray, 1000, 0.001, 5, None, None, 2, useHarrisDetector = True, k = 0.00009)
        if len(self.features.shape) == 3.0:
           assert(self.features.shape[1:] ==(1,2))
           self.features.shape = (self.features.shape[0], 2)

        for x, y in self.features:
           self.corners += [(x , y)]
           inter.append_features(x , y)
           cv2.circle(self.source_color, (x, y), 8, (255, 255, 255))

       	# subpixel accuracy
       	'''define the criteria to stop and refine the corners'''
    	criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.03)
    	#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    	cv2.cornerSubPix(gray, self.features, (5, 5), (-1, -1), criteria)
    
    	for x, y in self.features:
       		self.refined_corners += [(x , y)]
                inter.append_refined_features(x , y)
       		cv2.circle(self.source_color, (x, y), 4, (0, 255, 0))

    	'''Now draw them'''
    	res = np.hstack ((self.corners, self.refined_corners))
    	res = np.int0(res)
    	self.source_color[res[:,1], res[:, 0]] = [0, 0, 255]
    	self.source_color[res[:,3], res[:, 2]] = [0, 255, 0]
    	#cv2.imwrite(str(self.IMAGE_NAME) + '_subpixel.png', self.im_new)

        cv2.imwrite("images/result_images/Features and subpixel accuracy.jpg", self.source_color)
        #cv2.imshow("Features and subpixel accuracy", self.source_color)
        #cv2.waitKey(0)

        return inter


#   find the contours and the intersection points (using cv2.findContours, this method is not reliable for situation that 
#   one edges of plate are hidden or distrupted)
    def contourmethod(self, edges):
        hull = None
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.source_color, contours, -1, (0,0,255), 1)
        #cv2.imshow("contours", self.source_color)
        print hierarchy 
        c = max(contours, key = cv2.contourArea)
        cv2.drawContours(self.source_color, c, -1, (0,0,0), 3)
        #cv2.imshow("max_contour", self.source_color)
        cv2.waitKey(0)
        for i, cnt in enumerate(contours):
            if hierarchy[0,i,3] == -1 and cv2.contourArea(cnt) >= cv2.contourArea(c):
                hull = cv2.convexHull(cnt, returnPoints=True)
                break
        print hull 
        cv2.drawContours(self.source_color, [hull], -1, (255,255,0), 1)
        #cv2.imshow("hull", self.source_color)
        cv2.waitKey(0)
        length = len(hull)

        M = cv2.moments(hull)
        if M['m00'] == 0:
           M['m00'] = 0.01
        self.cx = int(M['m10']/M['m00'])
        self.cy = int(M['m01']/M['m00'])
        center = [self.cx, self.cy]
        #cv2.circle(self.source_color, (self.cx, self.cy), 5, (0, 255, 255), 2)
        
        print length
        coord = Coordinates()
        for i in xrange(0, length):
            if (i + 3) < length:
                [x, y] = coord.intersection((hull[i][0][0], hull[i][0][1]), (hull[i + 1][0][0], hull[i + 1][0][1]), (hull[i + 2][0][0], hull[i + 2][0][1]), (hull[i + 3][0][0], hull[i + 3][0][1]))
                coord.append(x, y)
                cv2.circle(self.source_color, (x, y), 6, (255, 255, 255), 2)
        #cv2.imshow("coordinates", self.source_color)
        #cv2.waitKey(0)

        return coord, center
    
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
        warpedImage = cv2.warpPerspective(source_2, transformationMatrix, (self.plate_size, self.plate_size))

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
        cv2.imwrite("images/result_images/warped_plate.jpg", sharpened)
        #cv2.imshow("sharpened", sharpened)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()


######################################################################################################################

class Perspective_plate(object):
    def __init__(self, source_color, platesize, platecolor_min, platecolor_max):
        self.source_color = source_color
        self.source_color = imutils.resize(self.source_color, width = 900)
        self.plate_size = platesize  #plate square size in milemeter
	self.plate_color_min = platecolor_min
	self.plate_color_max = platecolor_max
	self.source = source_color
	self.source_2 = source_color
	self.source = imutils.resize(self.source, width = 900)
	self.source_2 = imutils.resize(self.source_2, width = 900)
        #cv2.imshow("Original image", self.source_color)
	#cv2.waitKey(0)

    def houghlinemethod(self):
	self.source = cv2.GaussianBlur(self.source, (3, 3), 0)
	persp = Perspective(self.source, self.source_color, self.plate_size, self.plate_color_min, self.plate_color_max)
	edges = persp.handle()

	# HOUGH LINE TRANSFORM METHOD to compute the plate tranformation matrix
	lineinter = persp.houghlinemethod(edges)
        # check the intersections are quad
	if lineinter.quadcheck():
	    	centroid = lineinter.calculateCentroid()
	    	cv2.circle(self.source, (centroid[0], centroid[1]), 10, (0, 255, 0), 2)
	    	#cv2.imshow("Centroid", self.source)
	    	#cv2.waitKey(0)
	        corners = lineinter.calculateTRTLBRBL(centroid[0], centroid[1])               
	        print corners
	        print len(corners)
	        for i in xrange(0, len(corners)):
	            print corners[i]
	            cv2.circle(self.source, (corners[i][0][0], corners[i][0][1]), 6, (255, 105, 0), 2)
	        #cv2.imshow("Corners-Houghline method", self.source)
	        #cv2.waitKey(0)
	        warpedImage, transformationMatrix = persp.transform(corners, self.source_2)
	        persp.showsharpen(warpedImage)

        return warpedImage, transformationMatrix

    def houghlinePmethod(self):
	self.source = cv2.GaussianBlur(self.source, (5, 5), 0)
	persp = Perspective(self.source, self.source_color, self.plate_size, self.plate_color_min, self.plate_color_max)
	edges = persp.handle()

	# Probabilistic HOUGH LINE TRANSFORM METHOD to compute the plate tranformation matrix
	lineinter = persp.houghlinePmethod(edges)
        # check the intersections are quad
	if lineinter.quadcheck():
	    	centroid = (self.source_color.shape[1]/2, self.source_color.shape[0]/2)
	    	cv2.circle(self.source, (self.source_color.shape[1]/2, self.source_color.shape[0]/2), 10, (0, 255, 0), 2)
	    	#cv2.imshow("Centroid", self.source)
	    	#cv2.waitKey(0)
	        corners = lineinter.calculateTRTLBRBL(self.source_color.shape[1]/2, self.source_color.shape[0]/2)               
	        print corners
	        print len(corners)
	        for i in xrange(0, len(corners)):
	            print corners[i]
	            cv2.circle(self.source, (corners[i][0], corners[i][1]), 6, (255, 105, 0), 2)
	        cv2.imwrite("images/result_images/selected_corners.jpg", self.source)
	        #cv2.imshow("Corners-Probabilistic Houghline method", self.source)
        	#cv2.waitKey(0)
        	warpedImage, transformationMatrix = persp.transform(corners, self.source_2)
        	persp.showsharpen(warpedImage)
        return warpedImage, transformationMatrix

    def contourmethod(self):
	self.source = cv2.GaussianBlur(self.source, (3, 3), 0)
	persp = Perspective(self.source, self.source_color, self.plate_size, self.plate_color_min, self.plate_color_max)
	edges = persp.handle()

	# CONTOUR METHOD Probabilistic
	contourcoord, center_mass  = persp.contourmethod(edges)
        # check the intersections are quad
	if contourcoord.quadcheck():
	        centroid = contourcoord.calculateCentroid()
	        cv2.circle(self.source, (centroid[0], centroid[1]), 5, (0, 0, 0), 2)
	        #print center_mass
	        corners = contourcoord.calculateTRTLBRBL(centroid[0], centroid[1])
	        print corners
	        print len(corners)
	        for i in xrange(0, len(corners)):
	            print corners[i]
	            cv2.circle(self.source, (corners[i][0][0], corners[i][0][1]), 6, (0, 105, 255), 2)
	        #cv2.imshow("Corners-Contours metjod", self.source)
	        #cv2.waitKey(0)
	        warpedImage, transformationMatrix = persp.transform(corners)
	        persp.showsharpen(warpedImage)
        return warpedImage, transformationMatrix





