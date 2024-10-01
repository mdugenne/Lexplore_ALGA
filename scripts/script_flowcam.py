# Objective: This script was written to test a set of image processing steps on Python
import skimage as ski #pip install -U scikit-image
from skimage import color,measure,morphology
from skimage.measure import regionprops
from skimage.util import compare_images,crop
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
import copy
my_cmap = copy.copy(plt.colormaps.get_cmap('gray_r')) # get a copy of the gray color map
my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values

#Workflow starts here
path_to_network=Path("H:")
# Load raw and calibration images
outputfiles = list(Path(path_to_network / 'Flowcam_export').expanduser().rglob('*.tif'))
backgroundfiles,imagefiles=list(filter(lambda x: 'cal_' in x.name, outputfiles)),list(filter(lambda x: 'rawfile_' in x.name, outputfiles))
# Load metadata files (volume imaged, background pixel intensity, etc.)
metadatafiles=list(Path(path_to_network / 'Flowcam_export').expanduser().rglob('summary_*.csv'))
df_metadata=pd.concat(map(lambda file:pd.read_csv(file,sep=r'\,|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains("\:|\=",case=True)').assign(Name=lambda x: x.Name.str.replace(' ','_')).set_index('Name').T.rename(index={'Value':file.parent.name}),metadatafiles))
df_pixel=df_metadata[['Magnification','Calibration_Factor']]

# Load context file for cropping area
contextfiles=list(Path(path_to_network / 'Flowcam_export').expanduser().rglob('*.ctx'))
df_context=pd.concat(map(lambda file:pd.read_csv(file,sep=r'\=|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains("\[",case=True)').set_index('Name').T.rename(index={'Value':file.parent.name}),contextfiles))

cropping_area=list(df_context[['AcceptableTop','AcceptableBottom','AcceptableLeft','AcceptableRight']].values.flat)
file=imagefiles[538]
image = ski.io.imread(file,as_gray=True)[int(cropping_area[0]):int(cropping_area[1]),int(cropping_area[2]):int(cropping_area[3])]
background=ski.io.imread(file.parent / 'cal_image_000001.tif',as_gray=True)[int(cropping_area[0]):int(cropping_area[1]),int(cropping_area[2]):int(cropping_area[3])]
#Crop initial frame and correct for the calibration frame
plt.imshow(image, cmap='gray')
diff_image=compare_images(image,background, method='diff')
plt.figure()
plt.imshow(diff_image, cmap='gray')
plt.figure()
plt.imshow(ski.restoration.rolling_ball(diff_image,radius=int(df_context.FrameCount.values[0])), cmap='gray')
#Apply thresholding defined in the context file
markers = np.zeros_like(diff_image)
markers[diff_image >1/(df_context.ThresholdDark.astype(float).values[0])] = 1
plt.imshow(markers)

# Closing holes iteration and finding largest object
pixel_size=df_pixel.Calibration_Factor.str.strip(' ').astype(float).values[0] #in microns per pixel
filled_image=ski.morphology.remove_small_holes(ski.morphology.closing(markers,ski.morphology.square(int(np.ceil(int(df_context.DistanceToNeighbor.astype(float).values[0])/pixel_size)))).astype(bool))
plt.figure()
plt.imshow(filled_image, cmap='gray')

label_objects, nb_labels = sp.ndimage.label(filled_image)
labelled = measure.label(ski.morphology.remove_small_holes(markers.astype(bool),connectivity=1))
plt.figure()
plt.imshow(labelled, cmap='gray_r')

largest_object = labelled == np.argmax(np.bincount(labelled.flat)[1:])+1
plt.imshow(largest_object, cmap='gray_r')
# Measure properties
df_properties=pd.DataFrame(ski.measure.regionprops_table(label_image=largest_object.astype(int),intensity_image=image,properties=['area','area_bbox','area_convex','area_filled','axis_major_length','axis_minor_length','axis_major_length','bbox','centroid_local','centroid_weighted_local','eccentricity','equivalent_diameter_area','extent','image_intensity','inertia_tensor','inertia_tensor_eigvals','intensity_mean','intensity_max','intensity_min','intensity_std','moments','moments_central','moments_hu','num_pixels','orientation','perimeter','slice']),index=[file.parent.name])
# Generate and save thumbnails
contour = ski.morphology.dilation(largest_object.astype(int),footprint=ski.morphology.square(3), out=None, shift_x=False, shift_y=False)
contour -= largest_object.astype(int)
plt.imshow(np.pad(contour[df_properties.slice.values[0]]),cmap='gray_r')
plt.imshow(np.pad(image[df_properties.slice.values[0]],25),cmap='gray',alpha=0.2)


