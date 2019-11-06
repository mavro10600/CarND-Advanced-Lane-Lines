#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:36:49 2019

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

#%%
hls=cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel=hls[:, :, 2]
gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
sobelx=cv2.Sobel(gray, cv2.CV_64F, 1, 0)
abs_sobelx=np.absolute(sobelx)
scaled_sobel= np.uint8(255*abs_sobelx/np.max(abs_sobelx))
thresh_min=20
thresh_max=100
sxbinary=np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel<thresh_max)]=1
plt.imshow(sxbinary)

#%%
s_thresh_min=170
s_thresh_max=255
s_binary=np.zeros_like(s_channel)
s_binary[(s_channel>=s_thresh_min) & (s_channel<s_thresh_max)]=1
color_binary=np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))*255
combined_binary=np.zeros_like(sxbinary)
combined_binary[(s_binary==1)|(sxbinary==1)]=1
plt.imshow(combined_binary)

#%%

f, (ax1, ax2)=plt.subplots(2, 1, figsize=(5, 10))
ax1.set_title=("stacked_thresholds")
ax1.imshow(color_binary)
ax2.set_title=("color_binary")
ax2.imshow(combined_binary)