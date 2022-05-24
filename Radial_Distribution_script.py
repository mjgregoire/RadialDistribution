#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created May 2022

@author: mjgregoire
"""

#this code requires images to be 1 channel (DAPI), sums
#there is a macro to do that on imageJ


#import modules
import os
import shutil
from pathlib import Path


import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import csv
import imageio as iio
import skimage
from skimage import data
from skimage import io
from skimage.filters import threshold_mean
from skimage import measure
from skimage import filters
from skimage.measure import regionprops
from scipy.spatial import distance

file_list = [f for f in os.listdir('.') if (os.path.isfile(f) and f.split('.')[-1] != 'py')]
print(file_list)

#load the images
for f in file_list:
	file_name = os.path.splitext(f)[0]
	print(file_name)
	df1 = pd.DataFrame([file_name], columns=['file_name'])
	df1.to_csv('df1.csv', mode='a+')
	dot_tif = ".tif"
	join_names = file_name + dot_tif
	images = skimage.io.imread(join_names)
	print("image loaded")
	print("the size of the image is:", images.shape)
	df2 = pd.DataFrame([images.shape], columns=['image length', 'image width'])
	df2.to_csv('df2.csv', mode='a+')

#determine the threshold automatically
	thresh = skimage.filters.thresholding.threshold_mean(images)
	binary = images > thresh #> used because imaged cells are lighter than background
	print("Image thresholded")

#get the center of mass
	labeled_foreground = (images > thresh).astype(int)
	properties = regionprops(labeled_foreground, images)
	center_of_mass = properties[0].centroid
	#weighted_center_of_mass = properties[0].weighted_centroid
	print("the center of mass is:", center_of_mass)
	df3 = pd.DataFrame([center_of_mass], columns=['center mass x coordinate', 'center mass y coordinate'])
	df3.to_csv('df3.csv', mode='a+')
	x_cent = (center_of_mass[1],)
	#print(x_cent)
	y_cent = (center_of_mass[0],)
	#print(y_cent)
	# Note the inverted coordinates because plt uses (x, y) while NumPy uses (row, column)

	#plot the center of mass on the binary
	#fig, ax = plt.subplots()
	#ax.imshow(binary, cmap=plt.cm.gray)
	# Note the inverted coordinates because plt uses (x, y) while NumPy uses (row, column)
	#ax.scatter(center_of_mass[1], center_of_mass[0], s=160, c='C0', marker='+')
	#plt.savefig(file_name +'centerMass.png')

	
#find the area from the binary and get the radius of the cell if it was a perfect circle
	properties = measure.regionprops(labeled_foreground)
	area = [prop.area for prop in properties]
	#print(area)
	area = np.array(area)
	print("The area of the cell is:", area)
	df4 = pd.DataFrame([area], columns=["Area"])
	df4.to_csv('df4.csv', mode='a+')
	#[prop.perimeter for prop in properties] 
	pi = 3.14
	radius = float(math.sqrt(area / pi))
	print("The radius is:", radius)
	df5 = pd.DataFrame([radius], columns=["Radius"])
	df5.to_csv('df5.csv', mode='a+')

#set the values outside of the threshold a value of 0
	super_threshold_indices = images < thresh
	images[super_threshold_indices] = 0
	plt.imshow(images)

# plot the center of mass on the heatmap
	heatmap = px.imshow(images, color_continuous_scale='turbo')
	heatmap.add_trace(go.Scatter(x = x_cent, y = y_cent, mode='markers', marker=dict(size = 12, color='white', symbol = 'cross')))
	heatmap.write_html(file_name +'.html')

# calculate the 1D average of the image for each radius from the centroid position "center of mass"

	r, c = np.mgrid[0:images.shape[0], 0:images.shape[1]]
	# coordinates of origin
	O = [[84.67364503372877, 91.11211909746453]]
	# 2D array of pixel coordinates
	D = np.vstack((r.ravel(), c.ravel())).T

	metric = 'cityblock' # 'chebychev' #'euclidean'  #or 'cityblock' 
	# calculate distances
	dst = distance.cdist(O, D, metric)
	# group same distances
	dst_u, indices, total_count  = np.unique(dst, return_inverse=True,
                                         return_counts=True)
	# summed intensities for each unique distance
	f_image = images.flatten()
	proj_sum = [sum(f_image[indices == ix]) for ix, d in enumerate(dst_u)]
	# calculatge averaged pixel values
	projection = np.divide(proj_sum, total_count)

	plt.clf()
	plt.plot(projection)
	plt.xlabel('Distance[{}] from {}'.format(metric, O[0]))
	plt.ylabel('Averaged pixel value')
	plt.savefig(file_name +'radialDistribution.png')

#find max y and corresponding x at that index
	max_y = max(projection) 
	print("Max AVG pixel value is:", max_y)
	x_index = projection.argmax()
	print("Corresponding distance (cityblock, pixels) from origin is:", x_index)
	df6 = pd.DataFrame([max_y], columns=["Max AVG Pixel Value"])
	df7 = pd.DataFrame([x_index], columns=["Max AVG Pixel Value Corresponding Distance (cityblock)"])
	df6.to_csv('df6.csv', mode='a+')
	df7.to_csv('df7.csv', mode='a+')
	
#find the ratio of the peak to radius
	total_dist = (len(dst_u))
	df8 = pd.DataFrame([total_dist], columns=["Total Distance from Origin"])
	df8.to_csv('df8.csv', mode='a+')
	ratio_dist = x_index/total_dist
	print("The ratio of the distance of the peak divided by the total distance is:", ratio_dist)
	df9 = pd.DataFrame([ratio_dist], columns=["Ratio peak dist/total distance"])
	df9.to_csv('df9.csv', mode='a+')
	ratio = x_index/radius
	print("The ratio of the distance of the peak divided by the radius is:", ratio)
	df10 = pd.DataFrame([ratio], columns=["Ratio peak dist/radius"])
	df10.to_csv('df10.csv', mode='a+')


# Merge dfs
df_1 = pd.read_csv('df1.csv')
df_2 = pd.read_csv('df2.csv')
df_3 = pd.read_csv('df3.csv')
df_4 = pd.read_csv('df4.csv')
df_5 = pd.read_csv('df5.csv')
df_6 = pd.read_csv('df6.csv')
df_7 = pd.read_csv('df7.csv')
df_8 = pd.read_csv('df8.csv')
df_9 = pd.read_csv('df9.csv')
df_10 = pd.read_csv('df10.csv')
data = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10], axis=1, join="outer")
data = data.drop_duplicates()
data = data.drop(labels = 1, axis = 0)
data = data.drop(data.filter(regex='Unnamed').columns, axis=1)
print(data)
data.to_csv('radial_values.csv', mode='a+')
os.remove('df1.csv')
os.remove('df2.csv')
os.remove('df3.csv')
os.remove('df4.csv')
os.remove('df5.csv')
os.remove('df6.csv')
os.remove('df7.csv')
os.remove('df8.csv')
os.remove('df9.csv')
os.remove('df10.csv')
print("data saved")
