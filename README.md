# RadialDistribution

This is a python script that can be used to analyze radial distribution. 
It will analyze radial distribution from a central point of an ROI via the cityblock method (though this can easily be changed to Euclidean or Chebychev). 

The script works by finding a threshold to determine the ROI of the image, assigning any pixel values outside of the threshold a value of 0, finding the centroid of the ROI (values that pass the treshold), performing the radial distribution, plotting the values, and saving the outputs.

The script requires images to be .tif 1 channel (ex: DAPI -for chromatin radial distribution analysis) sums --No stacks. If images are in another format this can be easily changed in the code when you load in the images.

The script can be run by placing it as a .py file in the folder with your images to be analyzed. Open up a terminal and run the code by typing 
- `python Radial_Distribution_script.py`

The script will output to the command line as well as save a csv file including the image names, their size, the central position of the ROI, the max avg intensity value as well as it's corresponding distance, and the ratio of that distance to the total distance. 
It will also output a heatmap image with the centroid position mapped and the radial distribution plot as an interactive .html file.
