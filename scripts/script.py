# Objective: This script was written to test a set of image processing steps on Python
import skimage as ski #pip install -U scikit-image
import os
import scipy as sp
from pathlib import Path
import matplotlib.pyplot as mtp
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
outputfiles = list(Path('~\Documents\My CytoSense\Outputfiles').expanduser().rglob('*.jpg'))
image = ski.io.imread(outputfiles[-1],as_gray=True)

plt.imshow(image)

edges = ski.filters.sobel(image)
#Canny edge detection
edges=ski.feature.canny(image / 255.)
plt.imshow(edges)
# Fill holes and labels the individuals
fill_image = sp.ndimage.binary_fill_holes(edges)
plt.imshow(fill_image)
label_objects, nb_labels = sp.ndimage.label(fill_image)
#Watershed segmentation
elevation_map = ski.filters.sobel(image)
plt.imshow(elevation_map )
markers = np.zeros_like(image)
hist, hist_centers = ski.exposure.histogram(image)
markers[image < 30] = 1
markers[image > 150] = 2
plt.imshow(rgb2gray(image))