#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:57:47 2019

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

gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# si usas mpimg.imread())
#gray= cv2.cvtColor(img, cv2BGR2GRAY)#si usas cv2.imread())
sobelx= cv2.Sobel(gray, cv2.CV_64F,1,0)
sobely= cv2.Sobel(gray, cv2.CV_64F,0,1)
abs_sobelx=np.absolute(sobelx)
scaled_sobel=np.uint8(255*abs_sobelx/np.max(abs_sobelx))
plt.imshow (scaled_sobel)

#%%
thresh_min=20
thresh_max=100
sxbinary=np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel>thresh_min) & (scaled_sobel < thresh_max)]=1
plt.imshow(sxbinary)

#%%
R=img[:, :, 0]
G=img[:, :, 1]
B=img[:, :, 2]
thresh=(200, 255)
binary=np.zeros_like(R)
binary[(R > thresh[0])&(R <=thresh[1])]=1
plt.imshow(binary)

#%%
hls=cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
H=img[:, :, 0]
L=img[:, :, 1]
S=img[:, :, 2]
#%%
thresh=(90, 255)
binary=np.zeros_like(S)
binary[(S > thresh[0])&(S <=thresh[1])]=1
plt.imshow(binary)
#%%

thresh=(15, 100)
binary=np.zeros_like(H)
binary[(H > thresh[0])&(H <=thresh[1])]=1
plt.imshow(binary)
