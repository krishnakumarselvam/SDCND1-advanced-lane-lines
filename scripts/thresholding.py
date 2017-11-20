import cv2
import numpy as numpy

class ExtractLanes():
    def __init__(self, 
        abs_x_sobel_kernel=3, abs_x_thresh=(20, 100),
        abs_y_sobel_kernel=3, abs_y_thresh=(20, 100),
        mag_thresh_sobel_kernel=3, mag_thresh=(30, 100),
        dir_thresh_sobel_kernel=15, dir_thresh=(0.7, 1.3),
        color_s_thresh = (170, 255)
        ):

        self.abs_x_sobel_kernel = abs_x_sobel_kernel
        self.abs_x_thresh = abs_x_thresh

        self.abs_y_sobel_kernel = abs_y_sobel_kernel
        self.abs_y_thresh = abs_y_thresh

        self.mag_thresh_sobel_kernel = mag_thresh_sobel_kernel
        self.mag_thresh = mag_thresh

        self.dir_thresh_sobel_kernel = dir_thresh_sobel_kernel
        self.dir_thresh = dir_thresh
        self.color_s_thresh = color_s_thresh

    def extract_lanes(self, img):
                # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(img, orient='x', 
            sobel_kernel=self.abs_x_sobel_kernel, thresh=self.abs_x_thresh)
        grady = self.abs_sobel_thresh(img, orient='y', 
            sobel_kernel=self.abs_y_sobel_kernel, thresh=self.abs_y_thresh)
        mag_binary = self.mag_threshold(img, 
            sobel_kernel=self.mag_thresh_sobel_kernel, mag_thresh=self.mag_thresh)
        dir_binary = self.dir_threshold(img,
            sobel_kernel=self.dir_thresh_sobel_kernel, dir_thresh=self.dir_thresh)
        
        sxbinary = np.zeros_like(dir_binary)
        sxbinary[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        s_binary = self.color_threshold(img, color_s_thresh=self.color_s_thresh)

        combined_binary = np.zeros_like(dir_binary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
        return combined_binary

    def color_threshold(self, img, color_s_thresh):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= color_s_thresh[0]) & (s_channel <= color_s_thresh[1])] = 1
        return s_binary

    def abs_sobel_thresh(self, img, orient='x',sobel_kernel=3, thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # Return the result
        return binary_output

    def mag_threshold(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

        # Return the binary image
        return binary_output