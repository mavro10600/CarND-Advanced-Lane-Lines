#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:13:23 2019

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

img = mpimg.imread('../test_images/stopsign.jpg')
plt.imshow(img)

#%%
plt.imshow(img)
plt.plot(55, 70, '.')
plt.plot(150, 57, '.')
plt.plot(150, 100, '.')
plt.plot(55, 115, '.')
#%%
def warp(img):
    img_size=(img.shape[1], img.shape[0])
    src=np.float32([[55,70],[150,57],[150,100],[55,115]])
    dst=np.float32([[50,30],[120,30],[120,100],[50,100]])
    M=cv2.getPerspectiveTransform(src, dst)
    Minv=cv2.getPerspectiveTransform(dst, src)
    warped=cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

warped_img=warp(img)
f, (ax1, ax2) = plt.subplots(2,1,figsize=(20,10))
ax1.set_title('source image')
ax1.imshow(img)
ax2.set_title('dest image')
ax2.imshow(warped_img)