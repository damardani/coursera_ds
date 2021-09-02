# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:22:01 2021

@author: dwian
"""
from scipy import stats
import cv2
import glob
import numpy as np
import pandas as pd

image_list  = []
n_val = []
#Red
r_val = []
r_std = []
r_skw = []
#Green
g_val = []
g_std = []
g_skw = []
#Blue
b_val = []
b_std = []
b_skw = []

path = "D:\.ASkripsi\Data\Data_Gambar\Gambar\Merah\*.*"

def averagePixels(image):
    r, g, b = 0, 0, 0
    count = 0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            tempr, tempg, tempb = image[x, y]
            if (tempr and tempg and tempb)>0:
                r += tempr
                g += tempg
                b += tempb
                count += 1
            else:
                count += 0
    return (r/count), (g/count), (b/count), count

def Stdanskew(image):
    red   = []
    green = []
    blue  = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            tempr, tempg, tempb = image[x, y]
            if (tempr and tempg and tempb)>0:
                red.append(tempr)
                green.append(tempg)
                blue.append(tempb)
    
    #Red
    n_red    = len(red)
    mean_red = sum(red)/n_red
    var_red  = sum((r - mean_red)**2 for r in red)/(n_red)
    std_red  = var_red**(1/2)
    sk_red   = stats.skew(red)
    
    #Green
    n_green    = len(green)
    mean_green = sum(green)/n_green
    var_green  = sum((g - mean_green)**2 for g in green)/(n_green)
    std_green  = var_green**(1/2)
    sk_green   = stats.skew(green)
    
    #Blue
    n_blue     = len(blue)
    mean_blue  = sum(blue)/n_blue
    var_blue   = sum((b - mean_blue)**2 for b in blue)/(n_blue)
    std_blue   = var_blue**(1/2)
    sk_blue    = stats.skew(blue)
    
    return std_red, std_green, std_blue, sk_red, sk_green, sk_blue

for file in glob.glob(path):
    img = cv2.imread(file)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    first_mask  = hsv[:,:,2] >= 105
    second_mask = hsv[:,:,1] >= 200
    mask1 = first_mask*second_mask                 
    mask3 = mask1

    red   = rgb[:,:,0]*mask3
    green = rgb[:,:,1]*mask3
    blue  = rgb[:,:,2]*mask3
    masked = np.dstack((red, green, blue))
    image_list.append(masked)
    
for images in image_list:
    # Color Moment 1
    avg_r, avg_g, avg_b, count = averagePixels(images)
    r_val.append(avg_r)
    g_val.append(avg_g)
    b_val.append(avg_b)
    n_val.append(count)
    # Color Moment 2 dan 3
    std_red, std_green, std_blue, sk_red, sk_green, sk_blue = Stdanskew(images)
    r_std.append(std_red)
    g_std.append(std_green)
    b_std.append(std_blue)
    r_skw.append(sk_red)
    g_skw.append(sk_green)
    b_skw.append(sk_blue)
    
#CSV File
dict = {'red': r_val, 'green': g_val, 'blue': b_val,
        'red_std': r_std, 'green_std': g_std, 'blue_std': b_std,
        'red_skew': r_skw, 'green_skew': g_skw, 'blue_skew': b_skw,
        'npixel': n_val}
df   = pd.DataFrame(dict)
df.to_csv('RGB_Merah(NEW).csv')