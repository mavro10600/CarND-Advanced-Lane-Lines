## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image1]: ./output_images/undistort_test1.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "First Image"
[image3]: ./output_images/thresholded_color_test1.jpg "Binary Color Thresh"
[image4]: ./test_images/straight_lines1.jpg "Unwarp Example"
[image5]: ./output_images/straight_lines1.jpg "Warp Example"
[image6]: ./output_images/searching_points.png "Searching points"
[image7]: ./output_images/test2.jpg "Fit Visual"
[image8]: ./output_images/sliding_window.jpg "Sliding Window"
[image9]: ./output_images/final_window.jpg "Output"
[video1]: ./examples/output.avi "Video"
[video2]: ./project_video_processed_w_sliding_window.mp4 "Video Sliding"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in lines 22 through 62 of the file called `./examples/entregable.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 67 through 87 in `./examples/entregable.py`).  
First, the input image is passed through two filters, one, passed to HLS color space, second, a sobel x filter. Both resulting images are thresholded, on s channel in the HLS color space image, and the sobel one to get the closest to verticall derivative points.

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 102 through 109 in the file `./examples/entregable.py` .  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

I draw the points and their corresponding points on the image to get an idea of what to do. 
This code is on `./examples/pipeline.py`, lines 60 to 72. Pipeline is a program i used for test and calibration putposes on the testing images.
`
#%%
plt.imshow(result)
plt.plot(540, 450,'rs')
plt.plot(180, 700, 'rs')
plt.plot(1100, 700, 'rs')
plt.plot(740, 450, 'rs')

#%%
plt.imshow(result)
plt.plot(400, 100,'rs')
plt.plot(400, 500, 'rs')
plt.plot(900, 500, 'rs')
plt.plot(900, 100, 'rs')
`
![alt text][image6]

Afetr this, i affine through experimentation the correct src and dst coordinates.
This resulted in the following source and destination points (./examples/entregable.py lines 94 throgh 99):

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 540, 450      | 0, 0        | 
| 180, 700      | 360, 700      |
| 1100, 700     | 920, 700      |
| 740, 450      | 1280, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I defined four functions to accomplish this:

find_lane_pixels(binary_warped), defined in `./examples/entregable.py` in lines 116 throgh 192, where you get a binarized image already thresholded with sobel, HLS color space, and warped. First, applied a histogram function on the lower half of the image to found the x position of the lanes. Done this, it applies the sliding window algorithm to find the points which are mos likely to be part of the lane. The params used are nwindows=9, margin=100, minpix=50. Which basically say than it will search in 9 boxes through the image height, with a margin of 100 pixels of search of nonzero pixels, and that the minimum requirement of pixels to annotate than the lane is still there, are 50 pixels. It appends this data to the arrays leftx, lefty, rightx, righty. And draws on the image the boxes.

fit_polynomial_new(binary_warped), defined in `./examples/entregable.py` in lines 236 throgh 260. Where you input a binary thresholded image, and apply the previous functions to get the leftx, lefty, rightx, righty arrays which contains the coordinates of the lane points on the image. Then we apply `np.polyfit()` function to get the second order polynomial which fits to the data. These coefficients are stored in the variables left_fit and right_fit. Then, with this coefficients, we create a points space corresponding to the points in the image, and we plot the curves on the binarized image. At the end, this function returns a boolean value detected which says if the polynomial fit was succesful, the left_fit, right_fit coefficients, and the generated coordinates of the curve in the width space of the image. And the image with the curves painted on it.


The functions `search_around_poly_new(binary_warped, left_fit, right_fit)`, and `fit_poly_prior(img_shape, leftx, lefty, rightx,righty)` (lines 346 through 452 in `./examples/entregable.py` ) do the same stuff than the previous functions respectively, except than search_around_poly_new takes the previous line detection polynomial fit coefficients to recalculate the new set of points based on it. And the fit_poly_prior takes this point dataset, and calculates the new polynomial coefficients. And plots the identified curves and the masked area on the binary warped image.


![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 264 through 292 in my code in `./examples/entregable.py`. the function defined there is `measure_curvature_pixels_meters_new(ploty, left_fitx,right_fitx, ym_per_pix, xm_per_pix)` where ploty, left_fitx, right_fitx are the coordinate points of the lane pixels on the image. Based on them we create a second order polynomial fit which lives on the meters space, to do this, we use the ym_per_pix parameter, to convert the points data which lives in pixel space, to real world space in meters.

`
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
`

With this polynomial coefficients now we calculate the curvature associated to a point just in the bottom of the warped image, with this info, now we calculate the distance beetween both lines, and the position of the camera (and of the car) from the central line of them.
`
    #Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    lane_distance=((middlex-left_fitx[0])+(right_fitx[0]-middlex))*xm_per_pix
    #print('lane_distance: ',lane_distance)
    
    #left_distance=(middlex-left_fitx[0])*xm_per_pix
    #right_distance=(middlex-right_fitx[0])*xm_per_pix
    left_fit_x=(left_fit_cr[0]*(y_eval*ym_per_pix)**2+left_fit_cr[1]*(y_eval*ym_per_pix)+left_fit_cr[2])
    left_distance=(middlex*xm_per_pix-left_fit_x)
    right_fit_x=(right_fit_cr[0]*(y_eval*ym_per_pix)**2+right_fit_cr[1]*(y_eval*ym_per_pix)+right_fit_cr[2])
    right_distance=(-middlex*xm_per_pix+right_fit_x)
    if left_distance > right_distance:
        car_deviation=(lane_distance/2 -left_distance)*100
    elif right_distance > left_distance:
        car_deviation=(-lane_distance/2 +right_distance)*100
`



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 456 through 503 in my code in `./examples/entregable.py` in the function `process_image_prior()`.  Here is an example of my result on a test image:

![alt text][image8]
![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./examples/output.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It still has issues with the shadow of threes, if i imagine using this on a road beneath a forest, then i might be in troublesome. Either in a rainy situation or in darkness im not sure it has enough robustness.