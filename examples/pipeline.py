#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:27:01 2019

@author: mavro
"""

#%%
import sys
sys.path.remove ('/opt/ros/kinetic/lib/python2.7/dist-packages')
#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('../test_images/bridge_shadow.jpg')
plt.imshow(img)
s_binary=np.zeros_like(img)
sxbinary=np.zeros_like(img)
   
#%%

def pipeline(img,s_binary, sxbinary, s_thresh=(170, 255), sx_thresh=(20,100) ):
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

result, s_binary, sxbinary = pipeline (img, s_binary, sxbinary)

#PLot the result

f, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 20))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('original image', fontsize=40)
ax2.imshow(s_binary)
ax1.set_title('pipeline result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
print ('s_binary shape: ', s_binary.shape)
print ('sxbinary shape: ', sxbinary.shape)

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


#%%

src=np.float32([[580,430],[180,700],[1100,700],[780,430]])
#dst=np.float32([[400, 400],[400,700],[800,700],[800,400]])
dst=np.float32([[200, 0],[200,700],[1000,700],[1000,0]])
def warp(img, src, dst):
    img_size=(img.shape[1], img.shape[0])
    M=cv2.getPerspectiveTransform(src, dst)
    Minv=cv2.getPerspectiveTransform(dst, src)
    warped=cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

warped_img=warp(result, src, dst)
f, (ax1, ax2) = plt.subplots(2,1,figsize=(20,10))
ax1.set_title('source image')
ax1.imshow(result)
ax2.set_title('dest image')
ax2.imshow(warped_img)
#%%binary warped creation
binary_sum=np.add(sxbinary,s_binary)
binary_warped=warp(binary_sum, src, dst)
plt.imshow (binary_warped)
print('img_shape',binary_warped.shape)
plt.imsave('warped_example.jpg', binary_warped)
#%%
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#%%
# Load our image
# `mpimg.imread` will load .jpg as 0-255, so normalize back to 0-1
#img = mpimg.imread('warped_example.jpg')/255

def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = None

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = None
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    return histogram

# Create histogram of image binary activations
histogram1 = hist(s_binary)
histogram2 = hist(sxbinary)
# Visualize the resulting histogram
plt.plot(histogram2)



#%%
#FInding the histogram midpoints, and lines position or starting lines position
#print (warped_img.shape)
midpoint= np.int(histogram1.shape[0]//2)
print (midpoint)
midpoint= np.int(histogram2.shape[0]//2)
print (midpoint)

leftx_base=np.argmax(histogram1[:midpoint])
rightx_base=(np.argmax(histogram1[midpoint:])+midpoint)
print (leftx_base)
print (rightx_base)

leftx_base=((np.argmax(histogram1[:midpoint]))+(np.argmax(histogram2[:midpoint])))//2
rightx_base=((np.argmax(histogram1[midpoint:])+midpoint)+(np.argmax(histogram2[midpoint:])+midpoint))//2

#%%
#Sliding windows stuff

nwindows = 9 #number of sliding windows 
margin = 100 #width of windows  + / - margin
minpix=50 #minimum number of pixels found to recenter window
window_height=np.int(binary_warped.shape[0]//nwindows)
print ('window_height: ',window_height)
#identify the x and y positions of all nonzero pixels in the image
nonzero=binary_warped.nonzero()
nonzeroy=np.array(nonzero[0])
nonzerox=np.array(nonzero[1])
#cureent positions to be updated later for each window in nwindows
leftx_current=leftx_base
rightx_current=rightx_base
left_lane_inds=[]
right_lane_inds=[]


#%%
binary_jpg= mpimg.imread('warped_example.jpg')
#print('binare_warped_shape: ',binary_warped.shape)
#%%
def find_lane_pixels(binary_warped, binary_jpg):
    histogram= np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    plt.plot(histogram)
    #out_img=np.dstack((binary_warped, binary_warped, binary_warped))
    print('binary_warped.shape',binary_warped.shape)
    out_img=np.copy(binary_jpg)
    #out_img=binary_warped
    #plt.imshow(out_img)
    print('out_img.shape',out_img.shape)
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
    print('leftx_current',leftx_current)
    print('rightx_current',rightx_current)
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
        print('iterate: ',window)
        print('left:',win_y_low,' ',win_y_high,' ',win_xleft_high, ' ',win_xleft_low)
        print('right:',win_y_low,' ',win_y_high,' ',win_xright_high, ' ',win_xright_low)
        
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
print (binary_warped.shape)
leftx, lefty, rightx, righty, out_img=find_lane_pixels(binary_warped, binary_jpg)
plt.imshow(out_img)


#%%
def fit_polynomial(binary_warped, binary_jpg):
    leftx, lefty, rightx, righty, out_img=find_lane_pixels(binary_warped, binary_jpg)
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
    
    return out_img
out_img=fit_polynomial(binary_warped, binary_jpg)
plt.imshow(out_img)