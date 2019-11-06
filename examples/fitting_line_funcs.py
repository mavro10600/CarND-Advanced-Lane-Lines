#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:55:22 2019

@author: mavro
"""
#%%
import sys
sys.path.remove ('/opt/ros/kinetic/lib/python2.7/dist-packages')
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

#%%
binary_warped= mpimg.imread('warped_example.jpg')
print('binare_warped_shape: ',binary_warped.shape)
plt.imshow(binary_warped)

#%%
def find_lane_pixels(binary_warped):
    histogram= np.sum(binary_warped[binary_warped.shape[0]//2,:], axis=0)
    out_img=np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint=np.int(histogram.shape[0]//2)
    leftx_base=np.argmax(histogram[:midpoint])
    rightx_base=np.argmax(histogram[midpoint:])+midpoint
    
    nwindows=9
    margin = 100
    minpix=50
    
    window_height=np.int(binary_warped.shape[0]//nwindows)
    nonzero=binary_warped.nonzero()
    nonzeroy=np.array(nonzero[0])
    nonzerox=np.array(nonzero[1])
    leftx_current=leftx_base
    rightx_current=rightx_base
    
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

leftx, lefty, rightx, righty, out_img=find_lane_pixels(binary_warped)
plt.imshow(out_img[:,:,6:9])