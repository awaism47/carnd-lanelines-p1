#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I blur the image to reduce noise for the canny function. I use the low as 50 and high as 150 for thresholds to include bright pixels. I used the video/image dimensions when creating the masked region because I thought it might be good when image size is not known. My hough_lines function called up the draw_lines & Lane_lines function.  .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by making use of global variables and saving the data from previous frame to create a weighted average for the next. This allowed me to smooth the line and stop it shaking in the image. I obtained all the x1, x2, y1, y2 values and found an average then used it to calculate the slope. I defined the y1 and y2 points by using the image size and then used them to obtain the gradient of left and right lanes. This gradient was fed into another function called lane_lines. The lane_lines function used the data provided from the draw_line function and used it to find the y-intercept. This provides us with line equation which we could use to calculate the new weighted x-values. This stablises the line even if the car is turning.  ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the car turns because i found it hard to process the extra 'challenge' video because the lines were not able to keep up with curving lanes. Maybe it will require a second order function rather than just a linear curve.  ... 

Another shortcoming could be if the car pull in front within my lines, how would the process work, maybe i would need to define the colours more specifically.  ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to improve understanding of different processing spaces for opencv function, maybe grayscale is not helpful for changing road surfaces and colour.  ...

Another potential improvement could be to define the colours  and then it would be smoother for future roads with white and yellow lanes. ...