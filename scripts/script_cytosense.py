# Objective: This script performs a set of image processing steps on images acquired with the CytoSense imaging flow cytometer

# Load modules and functions required for image processing

try:
    from funcs_image_processing import *

except:
    from scripts.funcs_image_processing import *


# Workflow starts here:

pixel_size= cfg_metadata['pixel_size_cytosense'] #in microns per pixel
path_to_network=Path("R:") # Set working directory to forel-meco
outputfiles = list(Path(Path.home() / 'Documents'/'My CytoSense'/'Outputfiles').expanduser().rglob('*.jpg'))

exportfiles=natsorted(list(Path(path_to_network /'lexplore' / 'Lexplore' / 'export files' / 'IIF' ).expanduser().rglob('lexplore_lakewater_surface_smart*.zip')))#
map(lambda file: shutil.unpack_archive(file, file.parent / file.name.replace(' ','_').replace('.zip','')),exportfiles)  # Unzip export file
mosaicfiles=dict(map(lambda file: (file.name.split(' ')[0],natsorted(list(file.rglob('collage_*.png')))),runfiles))

backgroundfiles,imagefiles=list(filter(lambda x: '_background.jpg' in x.name, outputfiles)),list(filter(lambda x: ('_background.jpg' not in x.name) & ('_Cropped_' not in x.name), outputfiles))
file=imagefiles[1774]
df_properties=pd.DataFrame()

for file in natsorted(imagefiles):
    image = ski.io.imread(file,as_gray=True)
    background=ski.io.imread('_'.join(str(file).rsplit('_')[:-2])+'_background.jpg',as_gray=True)
    diff_image =compare_images(image,background, method='diff')
    #plt.imshow(diff_image, cmap='gray')

    markers = np.zeros_like(image)
    #markers[(background / image) > 255 + 7] = 1
    #markers[(background / image) < 255 - 7] = 1

    markers[diff_image >7/255] = 1
    labelled = measure.label(markers)
    #plt.imshow(labelled, cmap='gray' )

    largest_object = labelled == np.argmax(np.bincount(labelled.flat)[1:])+1
    #plt.imshow(largest_object, cmap='gray')

    '''
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
    
    labelled = measure.label(large_markers)
    largest_object = labelled == np.argmax(np.bincount(labelled.flat)[1:])+1
    plt.imshow(largest_object)
    '''
    # Measure properties
    df_properties=pd.concat([df_properties,pd.DataFrame(ski.measure.regionprops_table(label_image=largest_object.astype(int),intensity_image=image,properties=['area','area_bbox','area_convex','area_filled','axis_major_length','axis_minor_length','axis_major_length','bbox','centroid_local','centroid_weighted_local','eccentricity','equivalent_diameter_area','extent','image_intensity','inertia_tensor','inertia_tensor_eigvals','intensity_mean','intensity_max','intensity_min','intensity_std','moments','moments_central','moments_hu','num_pixels','orientation','perimeter','slice']),index=[file.name.split('_')[-1].split('.')[0]])],axis=0)

df_properties['Area']=df_properties['area']*(pixel_size**2)
df_properties['ECD']=2*np.sqrt(df_properties['Area']/np.pi)
df_properties['Biovolume']=(1/6)*np.pi*(df_properties['ECD']**3)
# Assign size bins and grouping index for each sample
df_properties = df_properties.assign(size_class=pd.cut(df_properties['ECD'], bins, include_lowest=True))
df_properties = pd.merge(df_properties, df_bins, how='left', on=['size_class'])
df_properties['volume']=4.3
df_nbss = df_properties.astype({'sizeClasses':str}).groupby(list(df_bins.columns)).apply(lambda x: pd.Series({'NBSS':sum(x.Biovolume ) / x.volume.unique()[0] / (x.range_size_bin.unique())[0]})).reset_index().sort_values('size_class_mid').reset_index(drop=True)
df_nbss,df_nbss_boot=nbss_estimates(df=df_properties.drop(columns=['Area','ECD','Biovolume']+list(df_bins.columns)).assign(Sample='_'.join(file.parent.name.split('_')[:-1])),pixel_size,grouping_factor=['Sample'])
plot = (ggplot(df_nbss) +
        #geom_point(mapping=aes(x='(1/6)*np.pi*(size_class_mid**3)', y='NBSS'), alpha=1) +  #
        geom_ribbon(mapping=aes(x='size_class_mid', y='NBSS',ymin='np.maximum(0,NBSS-NBSS_std)',ymax='NBSS+NBSS_std'),fill='red',alpha=0.1)+
        geom_point(mapping=aes(x='size_class_mid', y='NBSS'), alpha=1)+
        labs(x='Equivalent circular diameter ($\mu$m)',y='Normalized Biovolume Size Spectra ($\mu$m$^{3}$ mL$^{-1}$ $\mu$m$^{-3}$)', title='',colour='') +
        scale_y_log10(breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+04)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        scale_x_log10() +
        theme_paper).draw(show=False)

plot.savefig(fname='{}/figures/Initial_test/cytosense_test_nbss.png'.format(str(path_to_git)), dpi=300, bbox_inches='tight')


'''
# Generate and save thumbnails
contour = ski.morphology.dilation(largest_object.astype(int),footprint=ski.morphology.square(3), out=None, shift_x=False, shift_y=False)
contour -= largest_object.astype(int)

fig=plt.figure()
gs=matplotlib.gridspec.GridSpec(1,2)
ax1=fig.add_subplot(gs[0,0])
ax2=fig.add_subplot(gs[0,1])
thumbnail,ax=plt.subplots(2,1,figsize=(6,14), gridspec_kw={'height_ratios': [1,1]})
ax[0].imshow(np.pad(image.T,0,constant_values=float(background.mean()/255)),cmap='gray',alpha=0.7)
scale_value=500
scalebar=AnchoredSizeBar(transform=ax[0].transData,size=scale_value/pixel_size,
                           label='{} $\mu$m'.format(scale_value), loc='lower left',
                           pad=0.1,
                           color='white',
                           frameon=False,
                           fontproperties=fontprops)
ax[0].add_artist(scalebar)
ax[0].set_title('Raw image')
ax[1].imshow(np.pad((contour[df_properties.slice.values[0]]).T,50,constant_values=0),cmap='gray_r')
ax[1].imshow(np.pad((image[df_properties.slice.values[0]]).T,50,constant_values=background.mean()/255),cmap='gray',alpha=0.7)
scale_value=50
scalebar=AnchoredSizeBar(transform=ax[1].transData,size=scale_value/pixel_size,
                           label='{} $\mu$m'.format(scale_value), loc='lower left',
                           pad=0.1,
                           color='white',
                           frameon=False,
                           fontproperties=fontprops)
ax[1].add_artist(scalebar)
ax[1].set_title('Thumbnail')

thumbnail.savefig(fname=str( file.parent/ 'thumbnail_{}'.format(str(file.name.split('_')[-1]).rstrip())), bbox_inches="tight")

#Filtering smallest objects
markers[(markers==1) & (large_markers==0)] = 0
segmented_image = ski.segmentation.watershed(elevation_map, markers)
plt.imshow(segmented_image, cmap=plt.cm.gray)

plt.imshow(markers, cmap=plt.cm.nipy_spectral)

'''
