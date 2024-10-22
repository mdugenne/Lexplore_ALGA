# Objective: This script performs a set of image processing steps on images acquired with the CytoSense imaging flow cytometer

# Load modules and functions required for image processing

try:
    from funcs_image_processing import *

except:
    from scripts.funcs_image_processing import *


# Workflow starts here:

# Load camera resolution from configuration file and calibration file used to convert flow rate into volume imaged assuming sample core is a cylinder of width equivalent to the frame width and diameter indicated in the calibration file
pixel_size= cfg_metadata['pixel_size_cytosense'] #in microns per pixel
frame_width=cfg_metadata['width_cytosense'] #in pixels
df_calibration_flowrate=pd.read_csv(path_to_git / 'data' / 'Cytosense_calibration_flowrate_volume.csv')
reg_calibration=api.ols(formula='Cylinder_diameter~Flow_rate', data=df_calibration_flowrate).fit()

path_to_network=Path("R:{}".format(os.path.sep)) # Set working directory to forel-meco
outputfiles = list(Path(Path.home() / 'Documents'/'My CytoSense'/'Outputfiles').expanduser().rglob('*.jpg'))

exportfiles=natsorted(list(Path(path_to_network /'lexplore' / 'LeXPLORE' / 'export files' / 'IIF' ).expanduser().rglob('lexplore_lakewater_surface_smart*.zip')))#
path_to_images=list()
for file in exportfiles:
    path_to_images.append(Path(file.parent /file.name.replace(' ','_').replace('.zip','')).expanduser())
    if not Path().expanduser().exists():
        shutil.unpack_archive(file, file.parent / file.name.replace(' ','_').replace('.zip','')) # Unzip export file


imagefiles=dict(map(lambda file: ('_'.join(file.name.split('_')[:-1]),natsorted(list(file.rglob('*_Cropped_*.jpg')))),path_to_images))
#backgroundfiles,imagefiles=list(filter(lambda x: '_background.jpg' in x.name, outputfiles)),list(filter(lambda x: ('_background.jpg' not in x.name) & ('_Cropped_' not in x.name), outputfiles))
#file=imagefiles[1774]
df_properties=pd.DataFrame()
df_volume=pd.DataFrame()
df_nbss=pd.DataFrame()
for sample in list(imagefiles.keys()):
    # Append volume estimate and analysis duration
    path_to_sample_info=imagefiles[sample][0].parent.parent / str(imagefiles[sample][0].parent.parent.name.replace('Export_','')+'_Info.txt')
    path_to_sample_set=path_to_sample_info.parent / str('set_statistics_'+ path_to_sample_info.name.replace('_Info.txt','.csv'))
    if path_to_sample_info.expanduser().exists():
        if path_to_sample_set.expanduser().exists():
            df_sets=pd.read_table(path_to_sample_set,engine='python',encoding='utf-8',sep=r',')
            df_volume_sample=pd.read_table(path_to_sample_info,engine='python',encoding='utf-8',sep=r'\t',names=['Variable']).assign(Value=lambda x: x.Variable.str.split(':').str[1],Variable=lambda x: x.Variable.str.split(':').str[0]).set_index('Variable').T.rename(index={'Value':sample}).rename(columns={'Volume (μL)':'Volume_analyzed','Measurement duration':'Measurement_duration'})
            cytosense_processing_time_correction=df_sets.query('Set=="All Imaged Particles"').Count.values[0]/df_sets.query('Set=="IIF_large"').Count.values[0]
        else:
            cytosense_processing_time_correction =len(imagefiles[sample]) / float(df_volume_sample['Total number of particles'].values[0].lstrip())
        df_volume_sample=df_volume_sample.assign(Volume_analyzed=lambda x: x.Volume_analyzed.astype(float)*1e-03,Volume_pumped=lambda x: x['Flow rate (μL/sec)'].astype(float)*x.Measurement_duration.astype(float)*1e-03,Volume_Fluid_imaged=lambda x: np.pi*(reg_calibration.predict(pd.DataFrame({'Flow_rate':x['Flow rate (μL/sec)'].astype(float)}))*1e-03/2)*frame_width*pixel_size*1e-03*x.Measurement_duration.astype(float)*1e-03*(1+cytosense_processing_time_correction))
        df_volume = pd.concat([df_volume,df_volume_sample], axis=0)
    else:
        df_volume = pd.concat([df_volume,pd.DataFrame(index=[sample])],axis=0)
    with tqdm(desc='Generating vignettes for sample {}'.format(sample), total=len(natsorted(imagefiles[sample])), bar_format='{desc}{bar}', position=0, leave=True) as bar:
    images=[str(file) for file in imagefiles[sample]]
    for file in natsorted(images):
        file=Path(file).expanduser()
        percent = np.round(100 * (bar.n / len(images)), 1)
        bar.set_description('Generating vignettes for sample {} (%s%%)'.format(sample) % percent, refresh=True)
        image = ski.io.imread(file,as_gray=True)#,ski.io.imread(str(file).rsplit('_Cropped_')[0]+'_background.jpg',as_gray=True)
        #plt.figure(),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.show()

        #Segmentation
        edges = ski.filters.sobel(image)
        markers = np.zeros_like(image)
        markers[edges>np.quantile(edges,0.85)] = 1
        markers[(sp.ndimage.binary_closing(edges) ==False)] = 0

        #plt.imshow(markers, cmap='gray'),plt.show()
        fill_image = sp.ndimage.binary_fill_holes(markers)
        #plt.figure(),plt.imshow(fill_image),plt.show()
        label_objects, nb_labels = sp.ndimage.label(fill_image)
        # plt.figure(),plt.imshow(label_objects),plt.show()

        #Watershed segmentation
        elevation_map = ski.filters.sobel(image)
        #plt.imshow(elevation_map , cmap='gray')
        '''
        #Canny edge detection
        edges=ski.feature.canny(image,sigma=0.003 )
        #plt.imshow(edges, cmap='gray')
        #plt.figure()



        # Fill holes and labels the individuals
        fill_image = sp.ndimage.binary_fill_holes(edges)
        #plt.imshow(fill_image, cmap='gray')
        label_objects, nb_labels = sp.ndimage.label(fill_image)
        #plt.imshow(label_objects)



        # Filtering smallest objects
        markers = np.zeros_like(image)
        markers[image <  np.quantile(image,0.05)] = 1
        markers[image >=  np.quantile(image,0.99)] = 2
        label_objects, nb_labels = sp.ndimage.label(markers)
        #plt.imshow(label_objects)
        '''
        # Fix the lowest object size to 5 um
        largest_object_size=np.pi*(10*pixel_size/2)**2#np.quantile(np.sort(np.bincount(label_objects.flat))[::-2],q=0.9975)
        #largest_object_size=np.sort(np.bincount(label_objects.flat))[-2]
        large_markers = ski.morphology.remove_small_objects(label_objects, min_size=largest_object_size-1)
        #plt.figure(),plt.imshow(large_markers),plt.show()

        markers[(markers == 1) & (large_markers == 0)] = 0
        #plt.figure(),plt.imshow(markers, cmap=plt.cm.nipy_spectral),plt.show()
        fill_image = sp.ndimage.binary_fill_holes(markers)
        # plt.figure(),plt.imshow(fill_image),plt.show()
        label_markers, nb_labels = sp.ndimage.label(fill_image)
        # plt.figure(),plt.imshow(label_markers, cmap=plt.cm.gray),plt.show()

        segmented_image = ski.segmentation.watershed(elevation_map, markers)
        #plt.imshow(segmented_image, cmap=plt.cm.gray),plt.show()

        labelled = measure.label(large_markers)
        largest_object =np.isin(labelled, np.arange(0,len((np.bincount(labelled.flat)[1:])))+1) #labelled == np.argmax(np.bincount(labelled.flat)[1:])+1
        #plt.figure(),plt.imshow(largest_object),plt.show()

        # Measure properties
        particle_id=file.name.split('_')[-1].split('.')[0]
        df_properties_sample=pd.concat([pd.DataFrame({'Sample':sample,'nb_particles':nb_labels},index=[particle_id]),pd.DataFrame(ski.measure.regionprops_table(label_image=largest_object.astype(int),intensity_image=image,properties=['area','area_bbox','area_convex','area_filled','axis_major_length','axis_minor_length','axis_major_length','bbox','centroid_local','centroid_weighted_local','eccentricity','equivalent_diameter_area','extent','image_intensity','inertia_tensor','inertia_tensor_eigvals','intensity_mean','intensity_max','intensity_min','intensity_std','moments','moments_central','moments_hu','num_pixels','orientation','perimeter','slice']),index=[particle_id])],axis=1) if nb_labels>0 else pd.DataFrame({'Sample':sample,'nb_particles':nb_labels},index=[particle_id])
        df_properties=pd.concat([df_properties, df_properties_sample],axis=0)

        # Generate and save thumbnails for EcoTaxa
        contour = ski.morphology.dilation(largest_object.astype(int), footprint=ski.morphology.square(3), out=None, shift_x=False, shift_y=False)
        contour -= largest_object.astype(int)

        thumbnail, ax = plt.subplots(1, 1)
        scale_value=50
        ax.set_axis_off()
        padding=int(np.ceil(scale_value / pixel_size))
        ax.imshow(np.pad((contour), int(np.ceil(scale_value / pixel_size)), constant_values=0), cmap='gray_r')
        ax.imshow(cv2.cvtColor(ski.exposure.adjust_gamma(np.pad(image,padding , 'constant', constant_values=np.quantile(image,0.5)),0.7,gain=1), cv2.COLOR_BGR2RGB), alpha=.8)
        ax.imshow(cv2.cvtColor( ski.exposure.adjust_gamma(np.pad(image, padding, 'constant', constant_values=np.quantile(image, 0.5)), 0.7, gain=1), cv2.COLOR_BGR2RGB), alpha=.8)

        scalebar = AnchoredSizeBar(transform=ax.transData, size=scale_value / pixel_size,
                                   label='{} $\mu$m'.format(scale_value), loc='lower center',
                                   pad=0.1, color='black',frameon=False, fontproperties=fontprops)
        ax.add_artist(scalebar)
        #thumbnail.show()
        save_direcotry = Path(str(file.parent.parent).replace('export files{}IIF'.format(os.sep), 'ecotaxa').replace('Export_','').replace(' ','_')).expanduser()
        save_direcotry.mkdir(parents=True, exist_ok=True)
        thumbnail.savefig(fname=str( save_direcotry / 'thumbnail_{}_{}.png'.format(str(sample).rstrip(), str(particle_id).rstrip())),bbox_inches="tight")
        plt.close()
        bar.update(n=1)

    df_nbss_sample, df_nbss_boot_sample = nbss_estimates(df=pd.merge(df_properties.query('Sample=="{}"'.format(sample)),df_volume.assign(volume=lambda x: x.Volume_Fluid_imaged.astype(float))[['volume','Measurement_duration']],how='left',right_index=True,left_on=['Sample']), pixel_size=pixel_size, grouping_factor=['Sample'])
    df_nbss=pd.concat([df_nbss,df_nbss_sample],axis=0)



'''
df_properties['Area']=df_properties['area']*(pixel_size**2)
df_properties['ECD']=2*np.sqrt(df_properties['Area']/np.pi)
df_properties['Biovolume']=(1/6)*np.pi*(df_properties['ECD']**3)
# Assign size bins and grouping index for each sample
df_properties = df_properties.assign(size_class=pd.cut(df_properties['ECD'], bins, include_lowest=True))
df_properties = pd.merge(df_properties, df_bins, how='left', on=['size_class'])
df_properties['volume']=4.3
df_nbss = df_properties.astype({'sizeClasses':str}).groupby(list(df_bins.columns)).apply(lambda x: pd.Series({'NBSS':sum(x.Biovolume ) / x.volume.unique()[0] / (x.range_size_bin.unique())[0]})).reset_index().sort_values('size_class_mid').reset_index(drop=True)
df_nbss,df_nbss_boot=nbss_estimates(df=df_properties.assign(Sample='_'.join(file.parent.name.split('_')[:-1])),pixel_size=pixel_size,grouping_factor=['Sample'])
'''
plot = (ggplot(df_nbss) +
        #geom_point(mapping=aes(x='(1/6)*np.pi*(size_class_mid**3)', y='NBSS'), alpha=1) +  #
        #stat_summary(data=df_nbss_boot_sample.melt(id_vars=['Group_index','Sample','size_class_mid'],value_vars='NBSS'),mapping=aes(x='size_class_mid', y='value',group='Sample',fill='Sample'),geom='ribbon')+
        geom_ribbon(mapping=aes(x='size_class_mid', y='NBSS',ymin='np.maximum(0,NBSS-NBSS_std)',ymax='NBSS+NBSS_std',group='Sample',color='Sample'),alpha=0.1)+
        geom_point(mapping=aes(x='size_class_mid', y='NBSS',group='Group_index',colour='Sample'), alpha=1)+
        labs(x='Equivalent circular diameter ($\mu$m)',y='Normalized Biovolume Size Spectra ($\mu$m$^{3}$ mL$^{-1}$ $\mu$m$^{-3}$)', title='',colour='') +
        scale_y_log10(breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+04)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        scale_x_log10( breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e+00)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e+00)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(), labels=lambda l: ['1x10$^%s$' % round(np.floor(np.log10(v))) if ((v / (10 ** np.floor(np.log10(v)))) == 1) else '%s' % round( v / (10 ** np.floor(np.log10(v)))) for v in l]) +guides(colour=None)+
        theme_paper).draw(show=False)

plot.savefig(fname='{}/figures/Initial_test/cytosense_test_nbss.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')

'''
# Processing raw frames
        file=Path(file).expanduser()
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

'''

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


'''
'''
        # Generate and save thumbnails for EcoTaxa
        contour = ski.morphology.dilation(largest_object.astype(int), footprint=ski.morphology.square(4), out=None, shift_x=False, shift_y=False)
        contour -= largest_object.astype(int)
        contour = ski.measure.find_contours(measure.regionprops(largest_object.astype(int))[0].image, 0.5)[0]
        thumbnail, ax = plt.subplots(2, 1, figsize=(6, 14), gridspec_kw={'height_ratios': [1, 1]})
        ax[0].imshow(np.pad(image.T, 0, constant_values=float(background.mean() / 255)), cmap='gray', alpha=0.7)
        scale_value = 500
        scalebar = AnchoredSizeBar(transform=ax[0].transData, size=scale_value / pixel_size,
                                   label='{} $\mu$m'.format(scale_value), loc='lower left',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   fontproperties=fontprops)
        ax[0].add_artist(scalebar)
        ax[0].set_title('Raw image')
        ax[1].imshow(np.pad((contour[df_properties.slice.values[0]]).T, 50, constant_values=0), cmap='gray_r')
        ax[1].imshow(np.pad((image[df_properties.slice.values[0]]).T, 50, constant_values=np.quantile(image,0.5) ), cmap='gray', alpha=0.7)
        scale_value = 50
        scalebar = AnchoredSizeBar(transform=ax[1].transData, size=scale_value / pixel_size,
                                   label='{} $\mu$m'.format(scale_value), loc='lower left',
                                   pad=0.1, color='white',frameon=False, fontproperties=fontprops)
        ax[1].add_artist(scalebar)
        ax[1].set_title('Thumbnail')

        thumbnail.savefig(fname=str(file.parent / 'thumbnail_{}'.format(str(file.name.split('_')[-1]).rstrip())),bbox_inches="tight")

'''