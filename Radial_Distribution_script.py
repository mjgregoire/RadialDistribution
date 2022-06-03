#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created May 2022

@author: mjgregoire
"""

#this code requires images to be 1 channel (DAPI), sums
#there is a macro to do that on imageJ


## IMPORT MODULES
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

## GET THE LIST OF FILES FROM THE FOLDER

file_list = [f for f in os.listdir('.') if (os.path.isfile(f) and f.split('.')[-1] != 'py')]
print(file_list)

## LOAD THE IMAGES AND START THE FOR LOOP FOR THE SCRIPT
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

## DETERMINE IMAGE THRESHOLD AUTOMATICALLY
	thresh = skimage.filters.thresholding.threshold_mean(images)
	binary = images > thresh #> used because imaged cells are lighter than background
	print("Image thresholded")

## GET THE CENTER OF MASS FOR THE IMAGES
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

	## PLOT THE CENTER OF MASS ON THE BINARY
	#fig, ax = plt.subplots()
	#ax.imshow(binary, cmap=plt.cm.gray)
	# Note the inverted coordinates because plt uses (x, y) while NumPy uses (row, column)
	#ax.scatter(center_of_mass[1], center_of_mass[0], s=160, c='C0', marker='+')
	#plt.savefig(file_name +'centerMass.png')

	
## FIND AREA OF BINARY IMAGE AND GET RADIUS AS IF IT WAS PERECT CIRCLE
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

## SET THE VALUES OUTSIDE OF THE THRESHOLDED IMAGE A VALUE OF 0
	super_threshold_indices = images < thresh
	images[super_threshold_indices] = 0
	plt.imshow(images)

## PLOT THE IMAGE AS A HEATMAP WITH CENTER OF MASS AND MAX PIXEL INTENSITY
	#plot the center of mass 
	heatmap = px.imshow(images, color_continuous_scale='turbo')
	heatmap.add_trace(go.Scatter(x = x_cent, y = y_cent, mode='markers', marker=dict(size = 12, color='white', symbol = 'cross')))
	#find max pixel intensity and plot it
	point = np.unravel_index(images.argmax(), images.shape)
	#print(point)
	x_point = (point[1],)
	print("The x coordinate of the max intensity is:", x_point)
	df6 = pd.DataFrame([x_point], columns=["max x"])
	df6.to_csv('df6.csv', mode='a+')
	y_point = (point[0],)
	print("The y coordinate of the max intensity is:", y_point)
	df7 = pd.DataFrame([y_point], columns=["max y"])
	df7.to_csv('df7.csv', mode='a+')
	heatmap.add_trace(go.Scatter(x = x_point, y = y_point, mode='markers', marker=dict(size = 12, color='white', symbol = 'circle-open',)))
	#heatmap.show()
	heatmap.write_html(file_name +'.html')

## CALCULATE THE RADIAL DISTANCES FROM ORIGIN/CENTER
	# Get image parameters along the x and y and save to variables
	#print(images.shape)
	a = images.shape[0]
	b = images.shape[1]
	
	# Find radial distances corresponding to image center
	#create two grids — one corresponding to x-coordinates and one to y-coordinates
	[X, Y] = np.meshgrid(np.arange(b) - x_cent, np.arange(a) - y_cent) 
	#use the x and y values to calculate the radial distance at each point and make a new grid "R"
	#each point in this grid will have the value corresponding to the radial distance from the center
	R = np.sqrt(np.square(X) + np.square(Y))

## INTITIALIZE VARIABLES FOR THE AVERAGING CALCULATION
	#initialize array for x values 
	#set the resolution of averaging with array "rad" with radial values that go from 1 pixel (3rd arg) to the maximum radial value in grid
	rad = np.arange(1, np.max(R), 1)

	#initialize array for y values (intensity)
	#create an array of zeros with the same length as rad, and an index variable called index, which will be used to keep track of where we are changing the intensity
	intensity = np.zeros(len(rad))
	index = 0

## CALCULATE RADIAL AVERAGES
	#use for loop to calculate the average at each radial distance
	#set bin size (sets how many pixels we include in addition to the exact radial distance we are looking for)
	#here bin is 1 pixel less than and greater than our radius of interest

	bin_size = 1

	for i in rad:
    		mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
    		values = images[mask]
    		intensity[index] = np.mean(values)
    		index += 1
          
          
## MAKE PLOT OF AVG PIXEL RADIAL DISTRIBUTION
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(rad, intensity, linewidth=2)
	ax.set_xlabel('Radial Distance', labelpad=10)
	ax.set_ylabel('Average Intensity', labelpad=10)
	for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    		item.set_fontsize(8)
	plt.savefig(file_name +'radialDistribution.png')

	# save the intensity values to plot elsewhere
	intDF = pd.DataFrame([intensity])
	intDF.to_csv('intensityData.csv', mode='a+')

## FIND THE MAX AVERAGE RADIAL INTENSITY DISTANCE FROM CENTER 
	radmax_y = max(intensity)
	print("Max averaged radial intensity is:", radmax_y)
	df8 = pd.DataFrame([radmax_y], columns=["Max AVG radial intensity"])
	df8.to_csv('df8.csv', mode='a+')
	x_index = intensity.argmax()
	print("Corresponding radial distance (pixels) from origin is:", x_index)
	df9 = pd.DataFrame([x_index], columns=["Distance to max AVG radial intensity"])
	df9.to_csv('df9.csv', mode='a+')

	#normalize distance by the radius of the cell
	norm = x_index/radius
	print("The normalized distance (by radius) is:", norm)
	df10 = pd.DataFrame([norm], columns=["Normalized distance to max AVG radial intensity"])
	df10.to_csv('df10.csv', mode='a+')

	
## MERGE DFs
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
data.to_csv('radialDistributionData.csv', mode='a+')
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
