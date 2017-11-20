import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

class SmoothedLaneEquation:
    def __init__(self, memory_length = 8):
        self.frames_since_last_reject = 1000
        self.left_eqns = []
        self.right_eqns = []
        self.memory_length = memory_length

    def get_pct_change(self, existing_eqns, new_equation):
        combined_eqn = self.combine_eqns(existing_eqns)
        b = combined_eqn.coeff
        a = new_equation.coeff
        pct_change = np.mean((b-a)/a)*100
        return pct_change

    def process_new_data(self, left, right):
        PCT_THRESHOLD = 100
        if len(self.left_eqns) < 3 or abs(self.get_pct_change(self.left_eqns, left)) < PCT_THRESHOLD:
            self.left_eqns.append(left)
        else:
            self.left_eqns.pop(0)
        if len(self.right_eqns) < 3 or abs(self.get_pct_change(self.right_eqns, right)) < PCT_THRESHOLD:
            self.right_eqns.append(right)
        else:
            self.frames_since_last_reject
            self.right_eqns.pop(0)
        if len(self.left_eqns) > self.memory_length:
            self.left_eqns.pop(0)
        if len(self.right_eqns) > self.memory_length:
            self.right_eqns.pop(0)

    def combine_eqns(self, eqns):
        coeffs = [e.coeff for e in eqns]
        v = np.column_stack(coeffs)
        combined_coeffs = np.median(v, axis=1)
        return LaneEquation(combined_coeffs)

    def get_combined_eqns(self):
        return self.combine_eqns(self.left_eqns), self.combine_eqns(self.right_eqns)


class LaneEquation:
    def __init__(self, coeff, cov=None):
        self.coeff = coeff
        self.cov = cov

def fit_data(x, y, deg=1):
    fit = np.polyfit(x,y,deg, cov=True)
    return LaneEquation(fit[0], fit[1])

class ProcessFrame():
    def __init__(self, distortion_correcter, lane_extractor):
        self.distortion_correcter = distortion_correcter
        self.lane_extractor = lane_extractor
        self.smooth_equations = SmoothedLaneEquation()

        src = np.float32([[257, 719], [584, 464], [714, 464], [1162, 719]])
        dst = np.float32([[200, 700], [200, 0], [1000, 0], [1000, 700]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv= cv2.getPerspectiveTransform(dst, src)

    def fit_data(x, y, deg=1):
        fit = np.polyfit(x,y,deg, cov=True)
        return LaneEquation(fit[0], fit[1])

    def run(self, img):
        binary_warped = self.get_binary_warped_image(img)
        left_fitx, right_fitx, curvature, offset = self.get_lane_line_fit(binary_warped)
        return self.get_lane_fit_image(img, binary_warped, left_fitx, right_fitx, curvature, offset)

    def get_binary_warped_image(self, img):
        corrected_image = self.distortion_correcter.correct(img)
        binary = lane_extractor.extract_lanes(corrected_image)
        img_size = (img.shape[1], img.shape[0])
        binary_warped = cv2.warpPerspective(binary, self.M, img_size)
        return binary_warped

    def get_lane_line_fit(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Assuming you have created a warped binary image called "binary_warped"
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 


        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        left_fit = fit_data(lefty, leftx, deg=2)
        right_fit = fit_data(righty, rightx, deg=2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        max_plot_y = max(ploty)

        self.smooth_equations.process_new_data(left_fit, right_fit)
        if len(self.smooth_equations.left_eqns) > 1:
            left_fit, right_fit = self.smooth_equations.get_combined_eqns()
        
        left_fitx = np.polyval(left_fit.coeff, ploty)
        right_fitx = np.polyval(right_fit.coeff, ploty)

        y_eval = max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        curvature = (left_curverad + right_curverad)/2

        lane_center = (np.polyval(left_fit.coeff, max_plot_y) + np.polyval(right_fit.coeff, max_plot_y))/2
        image_center = binary_warped.shape[1]/2
        offset_m = xm_per_pix * (image_center - lane_center)

        return left_fitx, right_fitx, curvature, offset_m

    def get_lane_fit_image(self, img, binary_warped, left_fitx, right_fitx, curvature, offset):
        # Create an image to draw the lines on
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        _ = cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0])) 

        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        offset = int(offset * 100)
        font                   = cv2.FONT_HERSHEY_DUPLEX
        fontScale              = 1.25
        fontColor              = (0,0,0)
        lineType               = 2
        text1 = 'Curvature: {}m'.format(int(curvature))
        direction = 'left' if offset < 0 else 'right'
        text2 = 'Vehicle is {} cm to the {} of center'.format(abs(offset), direction)
        cv2.putText(result,text1, (100,100), font, fontScale, fontColor, lineType)
        cv2.putText(result,text2, (100,150), font, fontScale, fontColor, lineType)

        return result