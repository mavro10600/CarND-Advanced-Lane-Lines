#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 22:34:39 2019

@author: mavro
"""
#%%
import sys
sys.path.remove ('/opt/ros/kinetic/lib/python2.7/dist-packages')
#%%
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
#%%


# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    # 2) Convert to grayscale
    # 3) Find the chessboard corners
    # 4) If corners found: 
            # a) draw corners
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            # e) use cv2.warpPerspective() to warp your image to a top-down view
    #delete the next two lines
    #M = None
    #warped = np.copy(img) 
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    print (corners,' type:' ,type(corners))
    src=np.float32([[0,0]])
    counter=0
    y_counter=0
    print ('src type: ', type(src), src.shape)
    for x in corners:
        if counter==0:
            if y_counter==0 or y_counter==5:
                print (x[0], x.shape)
                src=np.append(src,x, axis=0)
        counter+=1
        if counter==8:
            if y_counter==0 or y_counter==5:
                print (x[0],'y_counter ',y_counter, x.shape)
                src= np.append(src,x, axis=0)
            y_counter+=1
            counter=0
    #src.reshape((2,5))
    src=np.delete(src, 0, axis=0)
    print ('src: ', src, 'shape: ', src.shape, src[3][1])        
    offset=100
    imgsize=(img.shape[1], img.shape[0])
    #dst=np.float32([[0,0],[src[1][0],0],[0,src[2][1]],[src[1][0],src[2][1]]])
    #dst=np.float32([[450,160],[1000,160],[450,750],[1000,750]])
    dst=np.float32([[offset,offset],[imgsize[0]-offset,offset],[offset,imgsize[1]-offset],[imgsize[0]-offset,imgsize[1]-offset]])
    print ('dst: ', dst)
    
   
    
    M=cv2.getPerspectiveTransform(src, dst)
    warped=cv2.warpPerspective(undist, M, imgsize, flags=cv2.INTER_LINEAR)
    return warped, M
#%%
# Read in an image
img = cv2.imread("../test_images/left_vel_ctrl.png")
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
#top_down= cv2.imread('undistort_output.png')
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
