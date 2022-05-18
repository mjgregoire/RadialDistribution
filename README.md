# RadialDistribution

This is a python script that can be used to analyze radial distribution. 

It requires images to be 1 channel (ex: DAPI -for chromatin radial distribution analysis) sums --No stacks.
The script can be run by placing the script in the folder with your images to be analyzed. Open up a terminal and run the code by typing 
`python Radial_Distribution_script.py`

The script will output to the command line as well as save a csv file including the image names, their size, and the central position of the ROI, as well as an image with the centroid position mapped onto the binary of the image, and the radial distribution plot.
