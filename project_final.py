#importing packages useful for this project
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np 
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio
import math
import sys, os
import os
#Create Global variables to save data from each frame and then use it to find a weighted average for gradient and y-intersect
global last_frame_right
global last_frame_left
last_frame_right=[]
last_frame_left=[]

#Function to convert the image from colour to gray

def convert_to_gray(img):
    #Applies the Grayscale transform
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#Function for canny algorithm 

def canny(img, low_threshold, high_threshold):
    #Applies canny transform
    return cv2.Canny(img,low_threshold, high_threshold)
#Function to blur image to reduce noise
def gaussian_blur(img, kernel_size):
    #Applies a gaussian noise kernel
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

#Function to mask the region of interest

def region_of_interest(img, vertices):
    #Applies an image masks and concentrates only on defined polygon
    mask = np.zeros_like(img)#Creates a blank image like the original

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape)>2:
        channel_count = img.shape[2] #i.e 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255

	#Filling pixels inside the polygon defined by 'vertices' with fill colour

    cv2.fillPoly(mask, vertices, ignore_mask_color)

	#returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image
#Draw lines using Hough lines image
def draw_lines(img, lines, color=[255,0,0], thickness=20):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below

    """
    #Create an empyty list for left and right hand lane
    global right_slope
    global right_y2
    global right_y1
    global right_x1
    global right_x2
    global left_slope
    global left_y2
    global left_y1
    global left_x1
    global left_x2    
    left_x1s=[]
    left_x2s=[]
    left_y1s=[]
    left_y2s=[]
    right_x1s=[]
    right_x2s=[]
    right_y1s=[]
    right_y2s=[]
    #Take the minimum of extroplation to be 0.65 of image size
    y_min1=img.shape[0]
    top=int(y_min1*0.65)
    bottom=int(img.shape[0])
    # Was encountering Nan's due to collecting values that were outside the lane 
    
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope == 0:
                continue
            if 0.8 > slope > 0.4:
            	left_x1s.append(x1)
            	left_x2s.append(x2)
            	left_y1s.append(y1)
            	left_y2s.append(y2)

            elif -0.8 < slope < -0.6:
            	right_x1s.append(x1)
            	right_x2s.append(x2)
            	right_y1s.append(y1)
            	right_y2s.append(y2)

    try:
    	avg_right_x1=int(np.mean(right_x1s))
    	avg_right_x2=int(np.mean(right_x2s))
    	avg_right_y1=int(np.mean(right_y1s))
    	avg_right_y2=int(np.mean(right_y2s))
    	right_slope=(avg_right_y2- avg_right_y1)/(avg_right_x2 - avg_right_x1)

    	#Linear interpolation
    	right_y1=top
    	right_x1=int(avg_right_x1 + (right_y1 - avg_right_y1)/right_slope)
    	right_y2=bottom
    	right_x2=int(avg_right_x2 + (right_y2 - avg_right_y2)/right_slope)

    except ValueError:
        pass

    try:
    	avg_left_x1=int(np.mean(left_x1s))
    	avg_left_x2=int(np.mean(left_x2s))
    	avg_left_y1=int(np.mean(left_y1s))
    	avg_left_y2=int(np.mean(left_y2s))
    	left_slope=(avg_left_y2- avg_left_y1)/(avg_left_x2 - avg_left_x1)
    	#Linear interpolation
    	left_y1=top
    	left_x1=int(avg_left_x1 + (left_y1 - avg_left_y1)/left_slope)
    	left_y2=bottom
    	left_x2=int(avg_left_x2 + (left_y2 - avg_left_y2)/left_slope)
    	# cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)	

    except ValueError:
        pass
    return lane_line(img, right_slope, left_slope, right_y2, right_y1, right_x1, right_x2, left_x2, left_x1, left_y2, left_y1)

#Function for hough transform
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    #returns an image with hough lines drawn
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    line_img=np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

#Function to take in the line data and use the previous frame to calculate weighted averages - Stablise the line
def lane_line(img, right_slope,left_slope, right_y2, right_y1, right_x1, right_x2, left_x2, left_x1, left_y2, left_y1,memory=0.9):
	m1=right_slope
	b1=right_y1-(m1*right_x1)
	m2=left_slope
	b2=left_y1-(m2*left_x1)
	global last_frame_right
	global last_frame_left
	if len(last_frame_right) == 0:
		last_frame_right=[int(m1), int(b1)]

	else:
		m1 = (last_frame_right[0]*memory)+(m1*(1-memory))
		b1 = (last_frame_right[1]*memory)+(b1*(1-memory))

	r_y1=right_y1
	r_x1=int((r_y1-b1)/m1)
	r_y2=right_y2
	r_x2=int((r_y2 - b1)/m1)
	last_frame_right = [m1,b1]
	
	cv2.line(img, (r_x1, r_y1), (r_x2, r_y2),  [255,0,0], 20)
	if len(last_frame_left) == 0:
		last_frame_left=[int(m2),int(b2)]
	else:
		m2 = (last_frame_left[0]*memory)+(m2 * (1-memory))
		b2 = (last_frame_left[1]*memory)+(b2*(1-memory))		
	l_y1=left_y1
	l_x1=int((l_y1-b2)/m2)
	l_y2=left_y2
	l_x2=int((l_y2-b2)/m2)
	last_frame_left = [m2,b2]

	cv2.line(img, (l_x1, l_y1), (l_x2, l_y2),  [255,0,0], 20)

	return last_frame_left, last_frame_right


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    #initial_img *α+img *β + λ
    return cv2.addWeighted(initial_img, α, img, β, λ)
def process_image(image):


    gray_image = convert_to_gray(image)
    #Blur the image to reduce noise
    blur_image=gaussian_blur(gray_image, 3)
    #Use canny edge detection to find edges
    low_threshold1=50
    high_threshold1=150
    canny_edge=canny(blur_image,low_threshold1,high_threshold1)
    #Mask region of interest
    #Grab the  and y coordinate from image
    ysize=image.shape[0]
    xsize=image.shape[1] #Note: always make a copy rather than simply using "="
    #
    left_bottom = [xsize*0,ysize]
    right_bottom = [xsize,ysize]
    apex = [xsize*0.5,ysize*0.58]
    vertices1=np.array([[(left_bottom),(apex),(apex),(right_bottom)]],dtype=np.int32)
    mask_image=region_of_interest(canny_edge,vertices1)
    #Define Hough transform parameters
    rho=1
    theta=np.pi/180
    threshold=1
    min_line_length = 25
    max_line_gap= 10
    lines=hough_lines(mask_image,rho,theta,threshold,min_line_length,max_line_gap)



    # you should return the final output (image where lines are drawn on lanes)

    return weighted_img(lines,image)
#Test into an image first
image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
new_image=process_image(image)
plt.imshow(new_image)
plt.show()
output1="Awais-SolidYellowLeft.mp4"
video=VideoFileClip("solidYellowLeft.mp4")
process_video=video.fl_image(process_image)
process_video.write_videofile(output1, audio=False)