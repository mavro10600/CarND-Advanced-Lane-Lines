#%%
import sys
sys.path.remove ('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#%%
import statistics as stat
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
%matplotlib qt
#%%
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print (type(objpoints))
print (len(objpoints))
print (len(imgpoints))
print (mtx)

#%%
def undistort(img,mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


#%%
#Create thresholded binary image

def pipeline(img, s_thresh=(170, 255), sx_thresh=(20,100) ):#170
    img=np.copy(img)
    #Convert HLS color sopace and separate the V channel
    hls=cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel=hls[:,:,1]
    s_channel=hls[:,:,2]
    
    sobelx=cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx=np.absolute(sobelx)
    scaled_sobel=np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    #Threshold x gradient
    sxbinary=np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0])&(scaled_sobel < sx_thresh[1])]=1
    #Threshold color channel
    s_binary=np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0])&(s_channel < s_thresh[1])]=1
    
    #stack each channel
    color_binary=np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))*255
    return color_binary, s_binary, sxbinary

#%%
    
# Apply a perspective transform to rectify binary image ("birds-eye view").

src=np.float32([[580,430],[180,700],[1100,700],[780,430]])
dst=np.float32([[200, 0],[200,700],[1000,700],[1000,0]])

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

#%%
    
#find lane pixels

def find_lane_pixels(binary_warped):
    histogram= np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    out_img=np.dstack((binary_warped, binary_warped, binary_warped))*255
    #print('binary_warped.shape',binary_warped.shape)
    
    #out_img=np.copy(binary_jpg)
    #out_img=binary_warped
    #plt.imshow(out_img)
    
    #print('out_img.shape',out_img.shape)
    
    #print('binary_warped',binary_warped)
    midpoint=np.int(histogram.shape[0]//2)
    leftx_base=np.argmax(histogram[:midpoint])
    rightx_base=np.argmax(histogram[midpoint:])+midpoint
    
    nwindows= 9
    margin = 100
    minpix=50
    
    window_height=np.int(binary_warped.shape[0]//nwindows)
    nonzero=binary_warped.nonzero()
    nonzeroy=np.array(nonzero[0])
    nonzerox=np.array(nonzero[1])
    leftx_current=leftx_base
    rightx_current=rightx_base
    
    #print('leftx_current',leftx_current)
    #print('rightx_current',rightx_current)
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    
    for window in range(nwindows):
        win_y_low=binary_warped.shape[0]-(window+1)*window_height
        win_y_high=binary_warped.shape[0]-window*window_height
        win_xleft_low=leftx_current-margin
        win_xleft_high=leftx_current+margin
        win_xright_low=rightx_current-margin
        win_xright_high=rightx_current+margin
        
        #print('iterate: ',window)
        #print('left:',win_y_low,' ',win_y_high,' ',win_xleft_high, ' ',win_xleft_low)
        #print('right:',win_y_low,' ',win_y_high,' ',win_xright_high, ' ',win_xright_low)
        
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),(0,255,0),2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),(0,255,0),2)
        
        good_left_inds=((nonzeroy>=win_y_low)&(nonzeroy<win_y_high) & (nonzerox>=win_xleft_low) &  (nonzerox<win_xleft_high)).nonzero()[0]
        good_right_inds=((nonzeroy>=win_y_low)&(nonzeroy<win_y_high) & (nonzerox>=win_xright_low) &  (nonzerox<win_xright_high)).nonzero()[0]
        
        #append indices to the list
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        #if you found > nminpix pixels, recenter each widnow on their new position
        
        if len(good_left_inds)> minpix:
            leftx_current=np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds)> minpix:
            rightx_current=np.int(np.mean(nonzerox[good_right_inds]))
        #concatenate the arays of indices 
    try:
        left_lane_inds=np.concatenate(left_lane_inds)
        right_lane_inds=np.concatenate(right_lane_inds)
    except ValueError:
        pass
    #extract right lane and left lane pixel positions
    leftx=nonzerox[left_lane_inds]
    lefty=nonzeroy[left_lane_inds]
    rightx=nonzerox[right_lane_inds]
    righty=nonzeroy[right_lane_inds]
        
    return leftx, lefty, rightx, righty, out_img


#%%
#Test pipeline so far

img = cv2.imread('../test_images/test1.jpg')
img_undist=undistort(img,mtx,dist)
plt.imshow(img_undist)

#%%

color_bin, s_bin, sx_bin= pipeline(img_undist)
plt.imshow(color_bin)

#%%
binary_sum=np.add(sx_bin,s_bin)
binary_warped=warper(binary_sum, src, dst)
plt.imshow (binary_warped)
print('img_shape',binary_warped.shape, 'binary_sum', binary_sum.shape)

#%%
leftx, lefty, rightx, righty, out_img=find_lane_pixels(binary_warped)
plt.imshow(out_img)
#%%

def fit_polynomial(binary_warped):
    leftx, lefty, rightx, righty, out_img=find_lane_pixels(binary_warped)
    #fit a secon order polynomial to each using no.polyfit
    left_fit=np.polyfit(lefty, leftx, 2)
    right_fit=np.polyfit(righty,rightx, 2)
    
    #Generate x and y values for plotting
    ploty= np.linspace(0, binary_warped.shape[0]-1,binary_warped.shape[0])
    try:
        left_fitx=left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        right_fitx=right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    except TypeError:
        print('the function failed to find a line')
        left_fitx=1*ploty**2+1*ploty
        right_fitx=1*ploty**2+1*ploty
    #visualization
    
    out_img[lefty,leftx]=[255,0,0]
    out_img[righty,rightx]=[0,0,255]
    
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    
    return ploty,left_fitx, right_fitx,out_img
ploty, left_fit, right_fit, out_img=fit_polynomial(binary_warped)
plt.imshow(out_img)


#%% Test pipeline on images 
images_test = glob.glob('../test_images/test*.jpg')
count=0
# Step through the list and search for chessboard corners
for fname in images_test:
    count+=1
    img = cv2.imread(fname)
    img_undist=undistort(img,mtx,dist)
    color_bin, s_bin, sx_bin= pipeline(img_undist)
    binary_sum=np.add(sx_bin,s_bin)
    binary_warped=warper(binary_sum, src, dst)
    ploty, left_fit, right_fit, out_img=fit_polynomial(binary_warped)
    plt.imsave('../output_images/test'+str(count)+'.jpg', out_img)


#%%
#Maybe src and dst arent quite good so far.
src=np.float32([[540,430],[180,700],[1100,700],[740,430]])
dst=np.float32([[0, 0],[360,700],[920,700],[1280,0]])

images_test = glob.glob('../test_images/straight_lines*.jpg')
count=0
# Step through the list and search for chessboard corners
for fname in images_test:
    count+=1
    img = cv2.imread(fname)
    img_undist=undistort(img,mtx,dist)
    binary_warped=warper(img_undist, src, dst)
    plt.imsave('../output_images/straight_lines'+str(count)+'.jpg', binary_warped)
    plt.imshow(binary_warped)

#%%
#Maybe src and dst arent quite good so far.
src=np.float32([[540,430],[180,700],[1100,700],[740,430]])
dst=np.float32([[0, 0],[360,700],[920,700],[1280,0]])

images_test = glob.glob('../test_images/test*.jpg')
count=0
# Step through the list and search for chessboard corners
for fname in images_test:
    count+=1
    img = cv2.imread(fname)
    img_undist=undistort(img,mtx,dist)
    binary_warped=warper(img_undist, src, dst)
    plt.imsave('../output_images/test_warped'+str(count)+'.jpg', binary_warped)
    plt.imshow(binary_warped)

#%% 
#Test individual images
s_thresh=(170, 255)
sx_thresh=(50,170)

src=np.float32([[540,450],[180,700],[1100,700],[740,450]])
dst=np.float32([[0, 0],[360,700],[920,700],[1280,0]])

img_4=cv2.imread('../test_images/test3.jpg')

img_undist=undistort(img_4,mtx,dist)
color_bin, s_bin, sx_bin= pipeline(img_undist, s_thresh, sx_thresh)
binary_sum=np.add(s_bin,sx_bin)
binary_warped=warper(binary_sum, src, dst)
ploty, left_fit, right_fit, out_img=fit_polynomial(binary_warped)

f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(10, 40))
f.tight_layout()
ax1.imshow(img_undist)
ax1.set_title('original image', fontsize=40)
ax2.imshow(binary_sum)
ax2.set_title('pipeline result', fontsize=40)
ax3.imshow(binary_warped)
ax3.set_title('pipeline result', fontsize=40)
ax4.imshow(out_img)
ax4.set_title('pipeline result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#%% Repeat_test_with pipeline
src=np.float32([[540,450],[180,700],[1100,700],[740,450]])
dst=np.float32([[0, 0],[360,700],[920,700],[1280,0]])

s_thresh=(170, 255)
sx_thresh=(50,170)

images_test = glob.glob('../test_images/test*.jpg')
count=0
# Step through the list and search for chessboard corners
for fname in images_test:
    count+=1
    print (fname)
    img = cv2.imread(fname)
    img_undist=undistort(img,mtx,dist)
    color_bin, s_bin, sx_bin= pipeline(img_undist, s_thresh, sx_thresh)
    binary_sum=np.add(s_bin,sx_bin)
    binary_warped=warper(binary_sum, src, dst)
    ploty, left_fit, right_fit, out_img=fit_polynomial(binary_warped)
    plt.imsave('../output_images/test'+str(count)+'.jpg', out_img)

#%%Function to calculate curvature radius
#%%Function to make the search form prior
def fit_poly_prior(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly_prior(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result

#%%
    
def measure_curvature_pixels(ploty, left_fitx, right_fitx):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    left_fit = np.polyfit(ploty, left_fitx, 2)
    right_fit = np.polyfit(ploty, right_fitx, 2)
    y_eval = np.max(ploty)
    print('y_eval: ', y_eval, 'left_fit[0]:', left_fit[0],'left_fit[1]:', left_fit[1], 'right_fit[0]', right_fit[0], 'right_fit[1]', right_fit[1])
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad


# Calculate the radius of curvature in pixels for both lane lines
#left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fit, right_fit)


#%% 
#Test individual images
s_thresh=(170, 255)
sx_thresh=(50,170)

src=np.float32([[540,450],[180,700],[1100,700],[740,450]])
dst=np.float32([[0, 0],[360,700],[920,700],[1280,0]])

img_4=cv2.imread('../test_images/test6.jpg')

img_undist=undistort(img_4,mtx,dist)
color_bin, s_bin, sx_bin= pipeline(img_undist, s_thresh, sx_thresh)
binary_sum=np.add(s_bin,sx_bin)
binary_warped=warper(binary_sum, src, dst)
ploty, left_fit, right_fit, out_img=fit_polynomial(binary_warped)
#print ('left_fit: ', left_fit, 'right_fit', right_fit)
# Calculate the radius of curvature in pixels for both lane lines
left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fit, right_fit)
print ('left_curverad: ', left_curverad, 'right_curverad: ', right_curverad)
f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(10, 40))
f.tight_layout()
ax1.imshow(img_undist)
ax1.set_title('original image', fontsize=40)
ax2.imshow(binary_sum)
ax2.set_title('pipeline result', fontsize=40)
ax3.imshow(binary_warped)
ax3.set_title('pipeline result', fontsize=40)
ax4.imshow(out_img)
ax4.set_title('pipeline result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#%%
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
middlex=binary_warped.shape[1]//2
print ('middlex',middlex)
def measure_curvature_pixels_meters(ploty, left_fitx, right_fitx, ym_per_pix, xm_per_pix):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
  
    y_eval = np.max(ploty)
    print('y_eval: ', y_eval, 'left_fit[0]:', left_fit[0],'left_fit[1]:', left_fit[1], 'right_fit[0]', right_fit[0], 'right_fit[1]', right_fit[1])
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    lane_distance=((middlex-left_fitx[0])+(right_fitx[0]-middlex))*xm_per_pix
    print('lane_distance: ',lane_distance)
    #left_distance=(middlex-left_fitx[0])*xm_per_pix
    #right_distance=(middlex-right_fitx[0])*xm_per_pix
    left_distance=(middlex*xm_per_pix-(left_fit_cr[0]*(y_eval*ym_per_pix)**2+left_fit_cr[1]*(y_eval*ym_per_pix)+left_fit_cr[2]))
    right_distance=(-middlex*xm_per_pix+(right_fit_cr[0]*(y_eval*ym_per_pix)**2+right_fit_cr[1]*(y_eval*ym_per_pix)+right_fit_cr[2]))
    if left_distance > right_distance:
        car_deviation=(lane_distance/2 -left_distance)*100
    elif right_distance > left_distance:
        car_deviation=(-lane_distance/2 +right_distance)*100
    print('left_distance', left_distance, 'right_distance: ', right_distance)
    #car_deviation=0
    return left_curverad, right_curverad, car_deviation
#%%
    
def weighted_img(img, initial_img,  α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img,α, img, β, γ )

#%% 
#Test individual images
s_thresh=(170, 255)
sx_thresh=(50,170)

src=np.float32([[540,450],[180,700],[1100,700],[740,450]])
dst=np.float32([[0, 0],[360,700],[920,700],[1280,0]])

img_4=cv2.imread('../test_images/test6.jpg')

img_undist=undistort(img_4,mtx,dist)
color_bin, s_bin, sx_bin= pipeline(img_undist, s_thresh, sx_thresh)
binary_sum=np.add(s_bin,sx_bin)
binary_warped=warper(binary_sum, src, dst)
ploty, left_fit, right_fit, out_img=fit_polynomial(binary_warped)
#print ('left_fit: ', left_fit, 'right_fit', right_fit)
# Calculate the radius of curvature in pixels for both lane lines
left_curverad, right_curverad, car_dev = measure_curvature_pixels_meters(ploty, left_fit, right_fit, ym_per_pix, xm_per_pix)
print ('left_curverad: ', left_curverad, 'right_curverad: ', right_curverad, 'car_deviation: ', car_dev)
unwarped=warper(out_img, dst, src)
result=weighted_img(img_undist,unwarped)

f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(10, 40))
f.tight_layout()
ax1.imshow(img_undist)
ax1.set_title('original image', fontsize=40)
ax2.imshow(binary_sum)
ax2.set_title('pipeline result', fontsize=40)
ax3.imshow(binary_warped)
ax3.set_title('pipeline result', fontsize=40)
ax4.imshow(result)
ax4.set_title('pipeline result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#%%

def fit_polynomial_new(binary_warped):
    leftx, lefty, rightx, righty, out_img=find_lane_pixels(binary_warped)
    #fit a secon order polynomial to each using no.polyfit
    left_fit=np.polyfit(lefty, leftx, 2)
    right_fit=np.polyfit(righty,rightx, 2)
    detected=True
    #Generate x and y values for plotting
    ploty= np.linspace(0, binary_warped.shape[0]-1,binary_warped.shape[0])
    try:
        left_fitx=left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        right_fitx=right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    except TypeError:
        print('the function failed to find a line')
        left_fitx=1*ploty**2+1*ploty
        right_fitx=1*ploty**2+1*ploty
        detected=False
    #visualization
    
    out_img[lefty,leftx]=[255,0,0]
    out_img[righty,rightx]=[0,0,255]
    
    #plt.plot(left_fitx, ploty, color='red')
    #plt.plot(right_fitx, ploty, color='red')
    
    return detected,ploty,left_fit,right_fit,left_fitx, right_fitx, out_img


#%%
def measure_curvature_pixels_meters_new(ploty, left_fitx, right_fitx, ym_per_pix, xm_per_pix):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    y_eval = np.max(ploty)
    #print('y_eval: ', y_eval, 'left_fit[0]:', left_fit[0],'left_fit[1]:', left_fit[1], 'right_fit[0]', right_fit[0], 'right_fit[1]', right_fit[1])
    
    # Calculation of R_curve (radius of curvature)
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
    #print('left_distance', left_distance, 'right_distance: ', right_distance)
    #car_deviation=0
    return left_fit_x,right_fit_x,left_fit_cr, right_fit_cr, left_curverad, right_curverad, car_deviation
#%%
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #coefficient values  of the last n fits of the line
        self.coefficients=[]
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def update(self, detected, recent_xfitted, current_fit, radius, line_base_pos, allx, ally ):
        self.detected=detected
        self.recent_xfitted.append(recent_xfitted)
        self.bestx=stat.mean(self.recent_xfitted)
        if len(self.coefficients) > 0:
            self.diffs=current_fit-self.coefficients[-1]
        else:
            self.diffs=0
        self.coefficients.append(current_fit)
        self.current_fit=current_fit
        self.best_fit=np.mean(self.coefficients)
        self.radius_of_curvature=radius
        self.line_base_pos=line_base_pos
        self.allx=allx
        self.ally=ally
        
        
#%%
from moviepy.editor import VideoFileClip
from moviepy.editor import CompositeVideoClip
#from IPyhton.display import HTML


#%%
def process_image(img, left, right):
    img_undist=undistort(img,mtx,dist)
    color_bin, s_bin, sx_bin= pipeline(img_undist, s_thresh, sx_thresh)
    binary_sum=np.add(s_bin,sx_bin)
    binary_warped=warper(binary_sum, src, dst)
    detected, ploty,left_ft, right_ft, left_fitx, right_fitx, out_img=fit_polynomial_new(binary_warped)
    #print ('left_fit: ', left_fit, 'right_fit', right_fit)
    # Calculate the radius of curvature in pixels for both lane lines
    left_fit_x,right_fit_x, left_fit, right_fit, left_curverad, right_curverad, car_dev= measure_curvature_pixels_meters_new(ploty, left_fitx, right_fitx, ym_per_pix, xm_per_pix)
    #print ('left_curverad: ', left_curverad, 'right_curverad: ', right_curverad, 'car_deviation: ', car_dev)
    unwarped=warper(out_img, dst, src)
    result=weighted_img(img_undist,unwarped)
    text='dev: '+str(car_dev/100)+' meters '+' curv: '+ str((left_curverad+right_curverad)/2)+' meters'
    font = cv2.FONT_HERSHEY_SIMPLEX   
    # org 
    org = (350,640)   
    # fontScale 
    fontScale = 3   
    # Blue color in BGR 
    color = (255, 0, 0)   
    # Line thickness of 2 px 
    thickness = 2   
    # Using cv2.putText() method 
    result = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA) 
    
    left.update(detected,left_fit_x ,left_ft,left_curverad, car_dev, left_fitx, ploty )
    
    return result

#%%
from moviepy.editor import *
# prints the maximum of red that is contained
# on the first line of each frame of the clip.
from moviepy.editor import VideoFileClip
myclip = VideoFileClip('../project_video.mp4').subclip(0,1)
print  ('size', myclip.size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))

#out=[]
left_line=Line()
right_line=Line()
for frame in myclip.iter_frames():
    out.write(process_image(frame, left_line, right_line))
    #out.write(frame)
out.release()


#%%Function to make the search form prior
def fit_poly_prior(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    try:
        print('try left_fit',lefty, leftx,'bad_right_fit: ', righty, rightx)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except TypeError:
        print('bad left_fit',lefty, leftx,'bad_right_fit: ', righty, rightx)
        left_fit=0
        right_fit=0
        detected=False
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    detected=True
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    try:
        left_fitx=left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        right_fitx=right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    except TypeError:
        print('the function failed to find a line')
        left_fitx=1*ploty**2+1*ploty
        right_fitx=1*ploty**2+1*ploty
        detected=False
    
    
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return detected,left_fit, right_fit, left_fitx, right_fitx, ploty

def search_around_poly_new(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 50
    
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    print('search_prior, left_fit:  ', left_fit, 'right_fit: ', right_fit, 'nonzero:', nonzero)
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    #right_lane_inds = ((nonzerox > 800))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    detected,left_ft, right_ft, left_fitx, right_fitx, ploty = fit_poly_prior(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,0, 255))
    cv2.fillPoly(window_img, np.int_([pts]), (255,0, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    
###warp_zero = np.zeros_like(warped).astype(np.uint8)
###color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
###pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
###pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
###pts = np.hstack((pts_left, pts_right))
###cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
###newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
###result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    
    
    return detected,ploty,left_ft, right_ft, left_fitx, right_fitx, result


#%%
def process_image_prior(img, left, right):
    img_undist=undistort(img,mtx,dist)
    color_bin, s_bin, sx_bin= pipeline(img_undist, s_thresh, sx_thresh)
    binary_sum=np.add(s_bin,sx_bin)
    binary_warped=warper(binary_sum, src, dst)
    print ('left_detected: ', left.detected, 'left_coeff_dim: ', len(left.coefficients))
    print ('right_detected: ', right.detected, 'right_coeff_dim: ', len(right.coefficients))
    if len(left.coefficients) > 1 :
        print('left_coff: ', left.coefficients[-1])
    if len(right.coefficients) > 1 :
        print('right_coff: ', right.coefficients[-1])
    if left.detected == False or right.detected == False:
        detected, ploty,left_ft, right_ft, left_fitx, right_fitx, out_img=fit_polynomial_new(binary_warped)
        print ('slifding_window')
    else:
        print('prior poly')
        detected, ploty, left_ft, right_ft, left_fitx, right_fitx, out_img=search_around_poly_new(binary_warped, left.coefficients[-1], right.coefficients[-1])
    #sanity check
    
    #print ('left_fit: ', left_fit, 'right_fit', right_fit)
    # Calculate the radius of curvature in pixels for both lane lines
    left_fit_x,right_fit_x, left_fit, right_fit, left_curverad, right_curverad, car_dev= measure_curvature_pixels_meters_new(ploty, left_fitx, right_fitx, ym_per_pix, xm_per_pix)
    #print ('left_curverad: ', left_curverad, 'right_curverad: ', right_curverad, 'car_deviation: ', car_dev)
    unwarped=warper(out_img, dst, src)
    result=weighted_img(img_undist,unwarped)
    if car_dev > 0:
        text1='dev: '+str(car_dev/100)+' meters to the left '
    elif car_dev <= 0:
        text1='dev: '+str(-car_dev/100)+' meters to the right '
    text2=' curv: '+ str((left_curverad+right_curverad)/2)+' meters'
    font = cv2.FONT_HERSHEY_SIMPLEX   
    # org 
    org1 = (300,280)   
    org2= (300, 350)
    # fontScale 
    fontScale = 2   
    # Blue color in BGR 
    color = (255, 0, 0)   
    # Line thickness of 2 px 
    thickness = 2   
    # Using cv2.putText() method 
    result=cv2.putText(result, text1, org1, font, fontScale, color, thickness, cv2.LINE_AA) 
    result=cv2.putText(result, text2, org2, font, fontScale, color, thickness, cv2.LINE_AA) 
    
    left.update(detected,left_fit_x ,left_ft,left_curverad, car_dev, left_fitx, ploty )
    right.update(detected,right_fit_x ,right_ft,right_curverad, car_dev, right_fitx, ploty )
    
    return result

#%%
from moviepy.editor import *
# prints the maximum of red that is contained
# on the first line of each frame of the clip.
from moviepy.editor import VideoFileClip
#myclip = VideoFileClip('../project_video.mp4').subclip(40.0,43.0)
myclip = VideoFileClip('../project_video.mp4')
print  ('size', myclip.size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))

#out=[]
left_line=Line()
right_line=Line()
for frame in myclip.iter_frames():
    out.write(process_image_prior(frame, left_line, right_line))
    #out.write(frame)
out.release()

