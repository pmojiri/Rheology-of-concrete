#!/usr/bin/python

####################################################################################
# File name : Entropy.py
# Author: Parisa Mojiri 
# Date created: 16/02/2016 
# Description: - Entropy class
#	       - skimage package is used in this class
#	       - inputs: image filename and selem size 
# 	       - output is a binary image after using entropy method
#
####################################################################################

# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np

import os
import cv2
import imutils

from skimage import data
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte, img_as_uint
from skimage import img_as_float
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import measure
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.exposure import equalize_hist
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.morphology import reconstruction
from skimage.exposure import rescale_intensity
################################################################################

class Entropy:

  # entropy
  def Obj_entropy_detection(self, filename, selem_size):

        self.selem_size = selem_size
        self.image =  filename
	#image = rescale_intensity(image, in_range=(50, 200))
	if self.image.shape[1] > 2500:
	    self.image = imutils.resize(self.image, width=2500)
	self.image = img_as_float(rgb2gray(self.image))
	#image = equalize_hist(image)


	self.image_entropy = entropy(self.image, disk(self.selem_size))
	#print type(self.image_entropy)
	#print self.image_entropy.shape
        #io.imsave('images/image_entropy.png', img_as_ubyte(self.image_entropy))

	thresh = threshold_otsu(self.image_entropy)
        thresh = 3
	self.image_binary = self.image_entropy > thresh
	seed = np.copy(self.image_binary)
	seed[1:-1, 1:-1] = self.image_binary.max()
	mask = self.image_binary
	self.image_filled = reconstruction(seed, mask, method='erosion')

	#image_binary = np.where(image_filled > np.mean(image_filled),1.0,0.0)

	self.image_morph = opening(self.image_binary, disk(4))
	self.image_morph = closing(self.image_binary, disk(4))
	io.imsave('images/image_binary_entropy.png', img_as_uint(self.image_morph))

	fig, axes  = plt.subplots(ncols=2, nrows =2, figsize=(10, 6), sharex=True,
                               sharey=True,
                               subplot_kw={"adjustable": "box-forced"})

	ax0, ax1, ax2, ax3 = axes.ravel()
	img0 = ax0.imshow(self.image, cmap=plt.cm.gray)
	ax0.set_title("Image")
	ax0.axis("off")
	fig.colorbar(img0, ax=ax0)

	img1 = ax1.imshow(self.image_entropy, cmap=plt.cm.gray)
	ax1.set_title("Entropy")
	ax1.axis("off")
	fig.colorbar(img1, ax=ax1)

	#for n, contour in enumerate(contours_1):    
	#    ax0.plot(contour[:, 1], contour[:, 0], linewidth=1)

	img2 = ax2.imshow(self.image_binary, cmap=plt.cm.gray)
	ax2.set_title("Binary")
	ax2.axis("off")
	fig.colorbar(img2, ax=ax2)

	#for n, contour in enumerate(contours_2):    
	#    ax2.plot(contour[:, 1], contour[:, 0], linewidth=1)

	img3 = ax3.imshow(self.image_morph, cmap=plt.cm.gray)
	ax3.set_title("Morph")
	ax3.axis("off")
	fig.colorbar(img3, ax=ax3)

	#for n, contour in enumerate(contours_3):    
	#    ax3.plot(contour[:, 1], contour[:, 0], linewidth=1)

	fig.tight_layout()
	#plt.show()
        return img_as_ubyte(self.image_binary)



