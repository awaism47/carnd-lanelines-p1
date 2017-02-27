import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
#read the image and print out some stats
image = mpimg.imread('test.jpg')
print('This image is: ', type(image), 'with dimensions:', image.shape)

#Grab the x and y size and make a copy of the image
ysize =image.shape[0]
xsize = image.shape[1]
#Note: always make a copy rather than simply using "="
line_image= np.copy(image)
color_select =np.copy(image)

#Define our colour criteria
red_threshold= 210
green_threshold =210
blue_threshold = 210
rgb_threshold =[red_threshold, green_threshold, blue_threshold]


#Define a triangle region of interest
# Keep in mind the origin x=0, y=0) is in the upper left in image processing
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in quiz
left_bottom = [125,539]
right_bottom = [850,539]
apex = [450, 300]

# Fit lines (y=Ax +B) to identify the 3 sided region of interest
# np.polyfit() returns the coefficients [A,B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]),(left_bottom[1],apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]),(left_bottom[1], right_bottom[1]),1)

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
# Mask color selection
color_select[color_thresholds] =[0,0,0]

# Color pixels red which are inside the region of interest
line_image[~color_thresholds & region_thresholds] = [255,0,0]

#Display the image
plt.imshow(color_select)
plt.imshow(line_image)
plt.show()