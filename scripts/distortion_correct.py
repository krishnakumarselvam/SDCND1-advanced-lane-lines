import cv2

class CorrectDistortion(object):
	def __init__(self, mtx, dist):
		self.mtx = mtx
		self.dist = dist

	def correct(self, img):
		undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
		return undist
