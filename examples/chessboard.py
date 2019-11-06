#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:44:09 2019

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

#%%
nx=8 
ny=6
fname = '../test_images/left_vel_ctrl.png'

img=cv2.imread(fname)
print (img)
print (type(img))
gray_1= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print ('gray shape',gray_1.shape)
print ('gray type',type(gray_1[0,0]))
print ('gray type',gray_1.dtype)
print (gray_1)
plt.imshow(gray_1)
ret, corners= cv2.findChessboardCorners(gray_1, (nx, ny), None)
if ret== True:
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    #plt.imshow(img)

#%%

img = mpimg.imread('../test_images/left_vel_ctrl.png')
img = mpimg.imread('../camera_cal/calibration6.jpg')

print('type', type(img))
#print (img)
img = img*255

print (img)
img = img.astype('uint8')
plt.imshow(img)
objpoints=[]
imgpoints=[]
objp=np.zeros((6*8,3),np.float32)
#print (img)
objp[:,:2]=np.mgrid[0:8, 0:6].T.reshape(-1,2)
#print ('objp',objp)
gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray=gray.astype('uint8')
print ('gray shape',gray.shape)
print ('gray type',type(gray[0,0]))
print ('gray type',gray.dtype)
print ('gray: ', gray)
print ('gray_1',gray_1)
plt.imshow(gray)
ret, corners=cv2.findChessboardCorners(gray, (8, 6), None)
print (ret)
if ret==True:
    imgpoints.append(corners)
    objpoints.append(objp)
    img=cv2.drawChessboardCorners(img, (8,6), corners, ret)
    plt.imshow(img)
#%%

img = mpimg.imread('../camera_cal/calibration1.jpg')
print('type', type(img))
#print (img)
img = img*255


img = img.astype('uint8')
print (img)

fname = '../camera_cal/calibration1.jpg'

img=cv2.imread(fname)

plt.imshow(img)
objpoints=[]
imgpoints=[]
objp=np.zeros((6*8,3),np.float32)
#print (img)
objp[:,:2]=np.mgrid[0:8, 0:6].T.reshape(-1,2)
#print ('objp',objp)
gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray=gray.astype('uint8')
print ('gray shape',gray.shape)
print ('gray type',type(gray[0,0]))
print ('gray type',gray.dtype)
print (gray)
plt.imshow(gray)
ret, corners=cv2.findChessboardCorners(gray, (8, 6), None)
print (ret)
#if ret==True:
#    imgpoints.append(corners)
#    objpoints.append(objp)
#    img=cv2.drawChessboardCorners(img, (8,6), corners, ret)
#    plt.imshow(img)