# Objective: This script was written to test a set of image processing steps on Python
import skimage as ski #pip install -U scikit-image
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy.core.defchararray import lower
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
import latex
matplotlib.rc('text', usetex = True)

import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
fontprops = fm.FontProperties(size=14,family='serif')
import copy
my_cmap = copy.copy(plt.colormaps.get_cmap('gray_r')) # get a copy of the gray color map
my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values
import warnings
warnings.filterwarnings("ignore")
from natsort import natsorted

#Workflow starts here
path_to_network=Path("R:") # Set working directory to forel-meco
# Load raw and calibration images
outputfiles = list(Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data' ).expanduser().rglob('Flowcam_10x_lexplore*/*.tif'))
backgroundfiles,imagefiles=list(filter(lambda x: 'cal_' in x.name, outputfiles)),list(filter(lambda x: 'rawfile_' in x.name, outputfiles))
# Load metadata files (volume imaged, background pixel intensity, etc.)
metadatafiles=list(Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data').expanduser().rglob('Flowcam_10x_lexplore*/*_summary.csv'))
df_metadata=pd.concat(map(lambda file:(df:=pd.read_csv(file,sep=r'\t',engine='python',encoding='latin-1',names=['Name','Value']),df:=df.Name.str.split(r'\,',n=1,expand=True).rename(columns={0:'Name',1:'Value'}),df:=df.drop(index=[0]).reset_index(drop=True).query('not Name.str.contains(r"\:|End",case=True)'),df:=df.assign(Name=np.where(df.Name.str.contains(r"\=",case=True).shift(1,fill_value=False),((df.loc[:5,'Name'].values.tolist()+(df.Name[3:-1].values+df.Name[4:].values).tolist())),df.Name)),df:=df.assign(Name=lambda x: x.Name.str.replace('========','').str.replace(' ','_').str.strip('_')).set_index('Name').T.rename(index={'Value':file.parent.name}))[-1],metadatafiles))
df_metadata=df_metadata.rename(columns={df_metadata.columns[-1]:'Metadata_Statistics_values','Filter_GridName':'Filter_Grid_Name'})
df_pixel=df_metadata[['Magnification','Calibration_Factor']]

# Load context file for cropping area
contextfiles=list(Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data' ).rglob('*.ctx'))
df_context=pd.concat(map(lambda file:pd.read_csv(file,sep=r'\=|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains("\[",case=True)').set_index('Name').T.rename(index={'Value':file.parent.name}),contextfiles))

df_cropping=df_context[['AcceptableTop','AcceptableBottom','AcceptableLeft','AcceptableRight']]
file=natsorted(imagefiles)[-1]#Path('R:Imaging_Flowcam/Flowcam data/Lexplore/Flowcam_10x_lexplore_wasam_20241002_2024-10-03 07-23-04/rawfile_011039.tif')#imagefiles[538]
sample_id=file.parent.name
cropping_area=df_cropping.loc['Lexplore']
colored_image=ski.io.imread(file,)[int(cropping_area[0]):int(cropping_area[1]),int(cropping_area[2]):int(cropping_area[3])]
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
markers[(background/image)>255+df_context.ThresholdDark.astype(float).values]=1
markers[(background/image)<255-df_context.ThresholdLight.astype(float).values]=1
markers[diff_image >1/(df_context[['ThresholdDark','ThresholdLight']].astype(float).min(axis=1).values[0])] = 1
plt.imshow(markers)

# Closing holes iteration and finding largest object
pixel_size=df_pixel.Calibration_Factor.str.strip(' ').astype(float) #in microns per pixel
filled_image=ski.morphology.remove_small_holes(ski.morphology.closing(markers,ski.morphology.square(int(np.ceil(int(df_context.DistanceToNeighbor.astype(float).values[0])/pixel_size.loc[sample_id])))).astype(bool))
filled_image=ski.morphology.closing(ski.morphology.dilation(ski.morphology.dilation(markers)))
plt.figure()
plt.imshow(filled_image, cmap='gray')

label_objects, nb_labels = sp.ndimage.label(filled_image)
labelled = measure.label(filled_image)#measure.label(ski.morphology.remove_small_holes(markers.astype(bool),connectivity=1))
plt.figure()
plt.imshow(labelled, cmap='gray_r')

2*np.sqrt(np.bincount(labelled.flat)*(0.7339**2)/np.pi)>float(df_context.MinESD.values[0])
largest_object = labelled == np.argmax(np.bincount(labelled.flat)[1:])+1
plt.imshow(largest_object, cmap='gray_r')
# Measure properties
df_properties=pd.DataFrame(ski.measure.regionprops_table(label_image=largest_object.astype(int),intensity_image=image,properties=['area','area_bbox','area_convex','area_filled','axis_major_length','axis_minor_length','axis_major_length','bbox','centroid_local','centroid_weighted_local','eccentricity','equivalent_diameter_area','extent','image_intensity','inertia_tensor','inertia_tensor_eigvals','intensity_mean','intensity_max','intensity_min','intensity_std','moments','moments_central','moments_hu','num_pixels','orientation','perimeter','slice']),index=[file.parent.name])
# Generate and save thumbnails
contour = ski.morphology.dilation(largest_object.astype(int),footprint=ski.morphology.square(3), out=None, shift_x=False, shift_y=False)
contour -= largest_object.astype(int)

fig=plt.figure()
gs=gridspec.GridSpec(1,2)
ax1=fig.add_subplot(gs[0,0])
ax2=fig.add_subplot(gs[0,1])
thumbnail,ax=plt.subplots(2,1,figsize=(6,14), gridspec_kw={'height_ratios': [1,1]})
ax[0].imshow(np.pad(image.T,0,constant_values=float(df_metadata.loc[sample_id,'Background_Intensity_Mean'])/255),cmap='gray',alpha=0.7)
scale_value=500
scalebar=AnchoredSizeBar(transform=ax[0].transData,size=scale_value/pixel_size.loc[sample_id],
                           label='{} $\mu$m'.format(scale_value), loc='lower left',
                           pad=0.1,
                           color='white',
                           frameon=False,
                           fontproperties=fontprops)
ax[0].add_artist(scalebar)
ax[0].set_title('Raw image')
ax[1].imshow(np.pad((contour[df_properties.slice.values[0]]).T,50,constant_values=0),cmap='gray_r')
ax[1].imshow(np.pad((image[df_properties.slice.values[0]]).T,50,constant_values=float(df_metadata.loc[sample_id,'Background_Intensity_Mean'])/255),cmap='gray',alpha=0.7)
scale_value=50
scalebar=AnchoredSizeBar(transform=ax[1].transData,size=scale_value/pixel_size.loc[sample_id],
                           label='{} $\mu$m'.format(scale_value), loc='lower left',
                           pad=0.1,
                           color='white',
                           frameon=False,
                           fontproperties=fontprops)
ax[1].add_artist(scalebar)
ax[1].set_title('Thumbnail')