#!/usr/bin/python

####################################################################################
# File name : intersections.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - Intersections class
#              - functions are defined for intersections	
#
####################################################################################
#
# import the necessary packages
import cv2
import numpy as np
import imutils
################################################################################

#coordinates of the number of intersections obtained (HOUGH LINE TRANSFORM METHOD)
class Intersections(object):
    inter = []
    features = []
    refined_features = []
    size = -1

    def __init__(self):
        Intersections.size += 1
        Intersections.centroidx = 0
        Intersections.centroidy = 0
        Intersections.sumx = 0
        Intersections.sumy = 0
        Intersections.corners = []
        Intersections.quad = 4
        self.points = []

    def centroidxy(self, x, y):
        Intersections.sumx += x
        Intersections.sumy += y        
         
#     append the coordinates of intersections
    def append(self, x, y):
        self.centroidxy(x, y)
        Intersections.inter.append([x, y])
        Intersections.size += 1

#     append the coordinates of detected features
    def append_features(self, x, y):
        Intersections.features.append([x, y])

#     append the refined coordinates of detected features
    def append_refined_features(self, x, y):
        Intersections.refined_features.append([x, y])

#     
    def closest_to_corners(self, x,y):
   	self.best_dist = 10e9
    	self.best_i = -1
    	for i in range(len(Intersections.refined_features)):
        	px = Intersections.refined_features[i][0]
        	py = Intersections.refined_features[i][1]
        	dx = (px-x)
        	dy = (py-y)
        	d = dx*dx + dy*dy
        	if d < self.best_dist:
            		self.best_dist = d
            		self.best_i = i
        return self.best_i
    
#     check if the points make up a quadrilateral    
    def quadcheck(self):
        Intersections.inter = np.reshape(Intersections.inter, (Intersections.size, 1, 2))
        print "Intersection.inter are :  " + str(Intersections.inter)
        #print type(Intersections.inter)
        #print type(Intersections.features)

        Intersections.inter = np.int32(Intersections.inter)
        Intersections.inter = cv2.convexHull(Intersections.inter)
        peri = cv2.arcLength(Intersections.inter, True)
        approx = cv2.approxPolyDP(Intersections.inter, 0.001 * peri, True)
        approx_list = approx.tolist()
        print approx_list
        for i in range(0, len(approx_list)):
          x = approx_list[i][0][0]
          y = approx_list[i][0][1]
          (x, y) = Intersections.refined_features[self.closest_to_corners(x,y)]
          self.points.append([x,y])

        d = []
        print "length approx list is "  + str (len(approx_list)) 
        if len(approx_list) != 4:
          for i in range(0, len(approx_list)):
              for j in range(i+1, len(approx_list)):
                  p1 = approx_list[i][0]
                  p2 = approx_list[j][0]
                  print p1, p2
                  print i, j
                  dist = np.sqrt(np.power((p2[0] - p1[0]),2) + np.power((p2[1] - p1[1]),2))
                  d.append(dist)
              print "d is :  " + str(d)    
              for k in range(0, len(d)):
                  if d[k] <= 100:
                      print k
                      #approx_list.remove(i)
                      approx_list.pop(k + 1)
                      print approx_list
                      #approx_list.append([(p2[0] + p1[0])/2, (p2[1] + p1[1])/2])
              d = []
              print "approx_list first iteration are" + str(approx_list)


        d = []
        features = Intersections.features
        if len(approx_list) != 4:
          for i in range(0, len(approx_list)):
              for j in range(0, len(features)):
                  p1 = approx_list[i][0]
                  p2 = features[j][0]
                  print p1, p2
                  print i, j
                  dist = np.sqrt(np.power((p2[0] - p1[0]),2) + np.power((p2[1] - p1[1]),2))
                  d.append(dist)
              print "d is :  " + str(d)    
              for k in range(0, len(d)):
                  if d[k] >= 100:
                      print k
                      #approx_list.remove(i)
                      approx_list.pop(k + 1)
                      print approx_list
                      #approx_list.append([(p2[0] + p1[0])/2, (p2[1] + p1[1])/2])
              d = []
              print "approx_list second iteration are"  + str(approx_list) 

        print self.points, "approx", len(self.points)
        if len(self.points) == Intersections.quad:
            print "yes a quad"
            Intersections.inter = self.points
            print Intersections.inter
            return True
        else:
            print "not a quad"
            Intersections.inter = self.points
            return False
        
#     find the centroid of the points of intersections found
    def calculateCentroid(self): 
        M = cv2.moments(np.array(Intersections.inter))
        print M
        if M['m00'] == 0:
           M['m00'] = 0.01
        Intersections.centroidx = int(M['m10']/M['m00'])
        Intersections.centroidy = int(M['m01']/M['m00'])
        print "centroids", Intersections.centroidx, Intersections.centroidy
        return [Intersections.centroidx, Intersections.centroidy]      
    
#     find the Top right, Top left, Bottom right and Bottom left points
    def calculateTRTLBRBL(self, cx, cy):
        topoints = []
        bottompoints = []
        cx = cx
        cy = cy
        print "coordinates   " + str(Intersections.inter)
        for inter in Intersections.inter:
            print inter, type(inter)
            print inter[1]
            print cy
            if inter[1] < cy:
                topoints.append(inter)
            else: 
                bottompoints.append(inter)
        print "top points" + str(topoints)
        print "bottom points" + str(bottompoints)
        top_left = min(topoints)
        top_right = max(topoints)
        bottom_right = max(bottompoints)
        bottom_left = min(bottompoints)
        
        Intersections.corners.append(top_left)
        Intersections.corners.append(top_right)
        Intersections.corners.append(bottom_right)
        Intersections.corners.append(bottom_left)
        print Intersections.corners
        return Intersections.corners

    def acceptLinePair(self, line1, line2, minTheta):
        theta1 = line1[1]
        theta2 = line2[1]
        #dealing with 0 and 180 ambiguities...
        if theta1 > np.pi : 
           theta1 += np.pi
        if theta2 > np.pi : 
           theta2 += np.pi
        return np.abs(theta1 - theta2) > minTheta

    def acceptLinesPPair(self, line1, line2, minTheta):
        theta1 = np.arctan2((line1[3] - line1[1]) , (line1[2] - line1[0]))
        theta2 = np.arctan2((line2[3] - line2[1]) , (line2[2] - line2[0]))
        #print theta1, theta2
        ##dealing with 0 and 180 ambiguities...
        #if theta1 > np.pi: 
        #   theta1 = theta1 + np.pi
        #if theta2 > np.pi: 
        #   theta2 = theta2 + np.pi
        #print theta1, theta2
        #print np.abs(theta1 - theta2) >  minTheta
        return  np.abs(theta1 - theta2) >  minTheta 
   
    def lineToPointPair(self, line):
        points = []
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        points.append([x1, y1])
        points.append([x2, y2])
        return points

#     find the intersection points of two lines         
    def computeintersect(self, line1, line2):
        P1 = self.lineToPointPair(line1)
        P2 = self.lineToPointPair(line2) 
        x1 = P1[0][0]
        y1 = P1[0][1]
        x2 = P1[1][0]
        y2 = P1[1][1]
        x3 = P2[0][0]
        y3 = P2[0][1]
        x4 = P2[1][0]
        y4 = P2[1][1]
        d = (((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
        #print d
        if d:
            inter_X = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / d
            inter_Y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / d
        
        #print "intersections",  inter_X, inter_Y
        return [inter_X, inter_Y] 

#     find the intersection points of two lines P        
    def computeintersectP(self, line1, line2):
        x1 = line1[0]
        y1 = line1[1]
        x2 = line1[2]
        y2 = line1[3]
        x3 = line2[0]
        y3 = line2[1]
        x4 = line2[2]
        y4 = line2[3]
        d = (((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
        #print d
        if d:
            inter_X = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / d
            inter_Y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / d
        
        print "intersections",  inter_X, inter_Y
        return [inter_X, inter_Y] 

