# USAGE
# python click_and_crop.py --image .jpg

# import the necessary packages
import cv2
import imutils

class click_and_crop:
  def __init__(self):
        # initialize the list of reference points and boolean indicating
        # whether cropping is being performed or not
        self.refPt = []
        self.cropping = False

  def click_crop(self, event, x, y, flags, param):
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		self.refPt = [(x, y)]
		self.cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		self.refPt.append((x, y))
		self.cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", self.image)

  def crop(self, image):
        # load the image, clone it, and setup the mouse callback function
        self.image = image
        self.image = imutils.resize(self.image, width = 800)
        clone = self.image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_crop)

        # keep looping until the 'q' key is pressed
        while True:
	    # display the image and wait for a keypress
	    cv2.imshow("image", self.image)
	    key = cv2.waitKey(1) & 0xFF

	    # if the 'r' key is pressed, reset the cropping region
	    if key == ord("r"):
		 self.image = clone.copy()

	    # if the 'c' key is pressed, break from the loop
	    elif key == ord("c"):
		 break

        # if there are two reference points, then crop the region of interest
        # from teh image and display it
        if len(self.refPt) == 2:
	    roi = clone[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
	    cv2.imshow("ROI", roi)
	    cv2.waitKey(0)

        # close all open windows
        cv2.destroyAllWindows()
        return roi
