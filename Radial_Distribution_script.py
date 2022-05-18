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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import imageio as iio
import skimage
from skimage import data
from skimage import io
from skimage.filters import threshold_mean
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
	df2 = pd.DataFrame([images.shape], columns=['length', 'width'])
	df2.to_csv('df2.csv', mode='a+')

#determine the threshold automatically
	thresh = skimage.filters.thresholding.threshold_mean(images)
	binary = images > thresh #> used because imaged cells are lighter than background
	print("Image thresholded")

#get the center of mass
	labeled_foreground = (images > thresh).astype(int)
	properties = regionprops(labeled_foreground, images)
	center_of_mass = properties[0].centroid
	weighted_center_of_mass = properties[0].weighted_centroid
	print("the center of mass is:", center_of_mass)
	df3 = pd.DataFrame([center_of_mass], columns=['center mass x coordinate', 'center mass y coordinate'])
	df3.to_csv('df3.csv', mode='a+')

#plot the center of mass
	fig, ax = plt.subplots()
	ax.imshow(binary, cmap=plt.cm.gray)
	# Note the inverted coordinates because plt uses (x, y) while NumPy uses (row, column)
	ax.scatter(center_of_mass[1], center_of_mass[0], s=160, c='C0', marker='+')
	plt.savefig(file_name +'centerMass.png')

# calculate the 1D average of the image for each radius from the centroid position "center of mass"

	r, c = np.mgrid[0:binary.shape[0], 0:binary.shape[1]]
	# coordinates of origin
	O = [[center_of_mass[1], center_of_mass[0]]]
	# 2D array of pixel coordinates
	D = np.vstack((r.ravel(), c.ravel())).T

	metric = 'chebychev' # 'chebychev' #'euclidean'  #or 'cityblock' 
	# calculate distances
	dst = distance.cdist(O, D, metric)
	# group same distances
	dst_u, indices, total_count  = np.unique(dst, return_inverse=True,
                                         return_counts=True)
	# summed intensities for each unique distance
	f_image = binary.flatten()
	proj_sum = [sum(f_image[indices == ix]) for ix, d in enumerate(dst_u)]
	# calculatge averaged pixel values
	projection = np.divide(proj_sum, total_count)

	plt.clf()
	plt.plot(projection)
	plt.xlabel('Distance[{}] from {}'.format(metric, O[0]))
	plt.ylabel('Averaged pixel value')
	plt.savefig(file_name +'radialDistribution.png')

# Merge dfs
df_1 = pd.read_csv('df1.csv')
df_2 = pd.read_csv('df2.csv')
df_3 = pd.read_csv('df3.csv')
data = pd.concat([df_1, df_2, df_3], axis=1, join="outer")
data = data.drop_duplicates()
data = data.drop(labels = 1, axis = 0)
data = data.drop(data.filter(regex='Unnamed').columns, axis=1)
print(data)
data.to_csv('radial_values.csv', mode='a+')
print("data saved")
