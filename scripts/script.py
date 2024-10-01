# Objective: This script was written to test a set of image processing steps on Python
import skimage as ski #pip install -U scikit-image
from skimage import color,measure,morphology
import os
import scipy as sp
from pathlib import Path
import matplotlib.pyplot as mtp
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
outputfiles = list(Path(Path.home() / 'Documents'/'My CytoSense'/'Outputfiles').expanduser().rglob('*.jpg'))

backgroundfiles,imagefiles=list(filter(lambda x: '_background.jpg' in x.name, outputfiles)),list(filter(lambda x: '_background.jpg' not in x.name, outputfiles))
file=imagefiles[1774]
image = ski.io.imread(file,as_gray=True)
background=ski.io.imread('_'.join(str(file).rsplit('_')[:-2])+'_background.jpg',as_gray=True)
plt.imshow(image-background, cmap='gray')
plt.figure()
plt.imshow(background, cmap='gray')
markers = np.zeros_like(image)
markers[image-background >7] = 1
labelled = measure.label(markers)
largestCC = labelled == np.argmax(np.bincount(labelled.flat)[1:])+1
rp = measure.regionprops(labelled)
[obj.area for obj in rp]

edges = ski.filters.sobel(image)

#Canny edge detection
edges=ski.feature.canny(image,sigma=0.003 )
plt.imshow(edges, cmap='gray')
plt.figure()

# Fill holes and labels the individuals
fill_image = sp.ndimage.binary_fill_holes(edges)
plt.imshow(fill_image, cmap='gray')
label_objects, nb_labels = sp.ndimage.label(fill_image)
plt.imshow(label_objects)
#Watershed segmentation
elevation_map = ski.filters.sobel(image)
plt.imshow(elevation_map , cmap='gray')

markers = np.zeros_like(image)
markers[image <  np.quantile(image,0.05)] = 1
markers[image >=  np.quantile(image,0.99)] = 2
label_objects, nb_labels = sp.ndimage.label(markers)
plt.imshow(label_objects)
largest_object_size=np.bincount(largestCC.flat)[1]
large_markers = ski.morphology.remove_small_objects(label_objects, min_size=largest_object_size/2)
plt.figure()
plt.imshow(large_markers)
#Filtering smallest objects
markers[(markers==1) & (large_markers==0)] = 0

plt.imshow(markers, cmap=plt.cm.nipy_spectral)
segmented_image = ski.segmentation.watershed(elevation_map, markers)
plt.imshow(segmented_image, cmap=plt.cm.gray)
