# Objective: This script performs a set of image processing steps on images acquired with the CytoSense imaging flow cytometer
from pickletools import uint8

import numpy as np
import pandas as pd
from plotnine import scale_shape

# Load modules and functions required for image processing

try:
    from funcs_image_processing import *

except:
    from scripts.funcs_image_processing import *

matplotlib.use('Qt5Agg')
# Workflow starts here:

# Load camera resolution from configuration file and calibration file used to convert flow rate into volume imaged assuming sample core is a cylinder of width equivalent to the frame width and diameter indicated in the calibration file
pixel_size= cfg_metadata['pixel_size_cytosense'] #in microns per pixel
frame_width=cfg_metadata['width_cytosense'] #in pixels
df_calibration_flowrate=pd.read_csv(path_to_git / 'data' / 'Cytosense_calibration_flowrate_volume.csv')
reg_calibration=api.ols(formula='Cylinder_diameter~Flow_rate', data=df_calibration_flowrate).fit()
if not Path('{}/figures/Initial_test/cytosense_calibration_flowrate.png'.format(str(path_to_git))).exists():
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=df_calibration_flowrate.dropna(subset=['Flow_rate','Cylinder_diameter']).Flow_rate,   y=df_calibration_flowrate.dropna(subset=['Flow_rate','Cylinder_diameter']).Cylinder_diameter)
    plot = (ggplot(data=df_calibration_flowrate.assign(diameter=lambda x: np.where(x.Cylinder_diameter.isna(),reg_calibration.predict(pd.DataFrame({'Flow_rate':x.Flow_rate})),x.Cylinder_diameter))) +
            geom_abline(slope=slope, intercept=intercept, alpha=1) +
            geom_point(aes(x='Flow_rate', y='diameter',shape='Cylinder_diameter.isna()'), size=5) +
            scale_shape_manual(limits=[True,False],values={True:"x",False:"o"})+
            labs(x='FLow rate ($\mu$L s$^{-1}$)', y=r'Sample core diameter ($\mu$m)') +
            scale_colour_manual(values={True:'#{:02x}{:02x}{:02x}{:02x}'.format(0, 0 , 0,10),False:'black'},guide=None)+
            annotate('text', label='y = ' + str(np.round( slope, 3)) + 'x ' + str( np.round(intercept, 2)) + ', R$^{2}$ = ' + str(np.round(r_value, 3)), x=np.nanquantile(df_calibration_flowrate['Flow_rate'], [0.999]), y=np.nanquantile(df_calibration_flowrate['Cylinder_diameter'], [0.999]), ha='right') +
            theme_paper+theme(axis_title=element_text(size=16),axis_text=element_text(size=16))).draw(show=False)
    plot.savefig(fname='{}/figures/Initial_test/cytosense_calibration_flowrate.png'.format(str(path_to_git)), dpi=300, bbox_inches='tight')


exportfiles=natsorted(list(Path(path_to_network /'lexplore' / 'LeXPLORE' / 'export files' / 'IIF' ).expanduser().rglob('lexplore_lakewater_surface_smart*_Images.zip')))#
path_to_images=list()
for file in exportfiles:
    if len(re.findall(r'_All Imaged Particles_|_IIF_',file.stem)):
        file.unlink(missing_ok=True) #Delete redundant image directories

    else:
        path_to_images.append(Path(file.parent /file.name.replace(' ','_').replace('.zip','')).expanduser())
        if not Path(file.parent / file.name.replace(' ','_').replace('.zip','')).expanduser().exists():
            try:
                shutil.unpack_archive(file, file.parent / file.name.replace(' ','_').replace('.zip','')) # Unzip export file
            except:
                print(r'Error uncompressing folder {}.\n Please check and re-export if needed'.format(str(file)))


imagefiles=dict(map(lambda file: ('_'.join(file.name.split('_')[:-1]),natsorted(list(file.rglob('*_Cropped_*.jpg')))),path_to_images))
#backgroundfiles,imagefiles=list(filter(lambda x: '_background.jpg' in x.name, outputfiles)),list(filter(lambda x: ('_background.jpg' not in x.name) & ('_Cropped_' not in x.name), outputfiles))
#file=imagefiles[1774]
df_properties=pd.DataFrame()
df_volume=pd.DataFrame()
df_nbss=pd.DataFrame()

for sample in list(imagefiles.keys())[::-1]:
    # Append volume estimate and analysis duration
    path_to_sample_info=imagefiles[sample][0].parent.parent / str(imagefiles[sample][0].parent.parent.name.replace('Export_','')+'_Info.txt')
    path_to_sample_set=path_to_sample_info.parent / str('set_statistics.csv')
    path_to_sample_listmode=Path(str(path_to_sample_info).replace('_Info.txt','_All Imaged Particles_Listmode.csv'))
    if path_to_sample_listmode.expanduser().exists():
        df_listmode = pd.read_csv(path_to_sample_listmode, sep = ',', encoding = 'utf-8').rename(columns=dict_cytosense_listmode)
    else:
        df_listmode=pd.DataFrame(dict(zip(['Particle ID']+list(dict_cytosense_listmode.values()),[pd.NA]*(len(dict_cytosense_listmode.values())+1))),index=[0])
    if path_to_sample_info.expanduser().exists():
        df_volume_sample = pd.read_table(path_to_sample_info, engine='python', encoding='utf-8', sep=r'\t',names=['Variable']).assign(Value=lambda x: x.Variable.astype(str).str.split('\:|\>').str[1:].str.join('').str.strip(),Variable=lambda x: x.Variable.str.split('\:|\>').str[0]).set_index('Variable').T.rename(index={'Value': sample}).rename( columns={'Volume (μL)': 'Volume_analyzed', 'Measurement duration': 'Measurement_duration'})
        if path_to_sample_set.expanduser().exists():
            df_sets=pd.read_table(path_to_sample_set,engine='python',encoding='utf-8',sep=r',')
            cytosense_processing_time_correction=df_sets.query('Set=="All Imaged Particles"').Count.values[0]/df_sets.query('Set=="IIF_large"').Count.values[0]
        else:
            cytosense_processing_time_correction =len(imagefiles[sample]) / float(df_volume_sample['Total number of particles'].values[0].lstrip())
        df_volume_sample=df_volume_sample.assign(Volume_analyzed=lambda x: x.Volume_analyzed.astype(float)*1e-03,Volume_pumped=lambda x: x['Flow rate (μL/sec)'].astype(float)*x.Measurement_duration.astype(float)*1e-03,Volume_Fluid_imaged=lambda x: np.pi*((reg_calibration.predict(pd.DataFrame({'Flow_rate':x['Flow rate (μL/sec)'].astype(float)}))*1e-03/2))*(frame_width*pixel_size*1e-03)*cfg_metadata['fps_cytosense']*x.Measurement_duration.astype(float)*1e-03).rename(columns={'Flow rate (μL/sec)':'Flow_rate'})#.assign(Volume_Fluid_imaged=lambda x:x.Volume_Fluid_imaged*cytosense_processing_time_correction*(x.Volume_analyzed/x.Volume_Fluid_imaged))
        df_volume = pd.concat([df_volume,df_volume_sample], axis=0)
    else:
        df_volume = pd.concat([df_volume,pd.DataFrame(index=[sample])],axis=0)
    # Loop through thumbnails
    images = [str(file) for file in imagefiles[sample]]
    with tqdm(desc='Generating vignettes for sample {}'.format(sample), total=len(natsorted(imagefiles[sample])), bar_format='{desc}{bar}', position=0, leave=True) as bar:

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

            #plt.figure(),plt.imshow(markers, cmap='gray'),plt.show()
            fill_image = sp.ndimage.binary_fill_holes(markers)
            #plt.figure(),plt.imshow(fill_image),plt.show()
            label_objects, nb_labels = sp.ndimage.label(fill_image)
            # plt.figure(),plt.imshow(label_objects),plt.show()

            #Watershed segmentation
            elevation_map = ski.filters.sobel(image)
            #plt.figure(),plt.imshow(elevation_map , cmap='gray')
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
            # Fix the lowest object size to 10 um
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
            if nb_labels>0:
                segmented_image = ski.segmentation.watershed(elevation_map, markers)
                #plt.figure(),plt.imshow(segmented_image, cmap=plt.cm.gray),plt.show()
                #image=(image*0.8).astype(np.uint8)


                labelled = measure.label(large_markers)
                largest_object =np.isin(labelled, np.arange(0,len((np.bincount(labelled.flat)[1:])))+1) #labelled == np.argmax(np.bincount(labelled.flat)[1:])+1
                #plt.figure(),plt.imshow(largest_object),plt.show()

                # Measure properties, save to EcoTaxa format, and append to existing samples
                particle_id=file.name.split('_')[-1].split('.')[0]
                df_properties_sample=pd.concat([pd.DataFrame({'img_file_name':'thumbnail_{}_{}.jpg'.format(str(sample).rstrip(), str(particle_id).rstrip()),'Sample':sample,'nb_particles':nb_labels},index=[particle_id]),pd.DataFrame(ski.measure.regionprops_table(label_image=largest_object.astype(int),intensity_image=image,properties=['area','area_bbox','area_convex','area_filled','axis_major_length','axis_minor_length','axis_major_length','bbox','centroid_local','centroid_weighted_local','eccentricity','equivalent_diameter_area','extent','image_intensity','inertia_tensor','inertia_tensor_eigvals','intensity_mean','intensity_max','intensity_min','intensity_std','moments','moments_central','num_pixels','orientation','perimeter','slice'],spacing=pixel_size),index=[particle_id])],axis=1) if nb_labels>0 else pd.DataFrame({'img_file_name':'thumbnail_{}_{}.png'.format(str(sample).rstrip(), str(particle_id).rstrip()),'Sample':sample,'nb_particles':nb_labels},index=[particle_id])
                df_properties=pd.concat([df_properties, df_properties_sample],axis=0).reset_index(drop=True)

                # Generate and save thumbnails for EcoTaxa
                contour = ski.morphology.dilation(largest_object.astype(int), footprint=ski.morphology.square(3), out=None, shift_x=False, shift_y=False)
                contour -= largest_object.astype(int)

                thumbnail, ax = plt.subplots(1, 1)
                scale_value=50
                ax.set_axis_off()
                padding=int(np.ceil(scale_value / pixel_size))
                #ax.imshow(np.pad((contour), int(np.ceil(scale_value / pixel_size)), constant_values=0), cmap='gray_r')
                ax.imshow(cv2.cvtColor(ski.exposure.adjust_gamma(np.pad(image,padding , 'constant', constant_values=np.quantile(image,0.5)),0.7,gain=1), cv2.COLOR_BGR2RGB), alpha=1)
                #ax.imshow(cv2.cvtColor( ski.exposure.adjust_gamma(np.pad(image, padding, 'constant', constant_values=np.quantile(image, 0.5)), 0.7, gain=1), cv2.COLOR_BGR2RGB), alpha=.8)

                scalebar = AnchoredSizeBar(transform=ax.transData, size=scale_value / pixel_size,
                                           label='{} $\mu$m'.format(scale_value), loc='lower center',
                                           pad=0.1, color='black',frameon=False, fontproperties=fontprops)
                ax.add_artist(scalebar)
                #thumbnail.show()
                save_directory = Path(str(file.parent.parent).replace('export files{}IIF'.format(os.sep), 'ecotaxa').replace('Export_','').replace(' ','_')).expanduser()
                save_directory.mkdir(parents=True, exist_ok=True)
                thumbnail.savefig(fname=str( save_directory / 'thumbnail_{}_{}.jpg'.format(str(sample).rstrip(), str(particle_id).rstrip())),bbox_inches="tight",dpi=300,pad_inches=0)
                plt.close()
            bar.update(n=1)
        # Generate abd save table for EcoTaxa
    filename_ecotaxa = str(save_directory /'ecotaxa_table_{}.tsv'.format(str(sample).rstrip()))
    df_ecotaxa = generate_ecotaxa_table(df=pd.merge(pd.merge(df_properties.query('Sample=="{}"'.format(sample)),df_volume.loc[sample].to_frame().T.assign(sample_trigger=lambda x: pd.Series(x.values[0])[pd.Series(x.values[0]).astype(str).str.contains('TRIGGER')].values[0].split('-')[0].strip().replace(' ','')).rename(columns={'Volume_analyzed':'sample_volume_analyzed_ml','Volume_pumped':'sample_volume_pumped_ml','Volume_Fluid_imaged':'sample_volume_fluid_imaged_ml','Trigger level (mV)':'sample_trigger_threshold_mv','Measurement_duration':'sample_duration_sec','Flow_rate':'sample_flow_rate'})[['sample_trigger','sample_volume_analyzed_ml','sample_volume_pumped_ml','sample_volume_fluid_imaged_ml','sample_trigger_threshold_mv','sample_duration_sec','sample_flow_rate']],how='left',right_index=True,left_on=['Sample']).assign(Particle_ID=lambda x: x.img_file_name.str.rsplit('_').str[-1].str.replace('.jpg','').astype(int)),df_listmode.astype({'Particle_ID':int}),how='left',on='Particle_ID'), instrument='CytoSense', path_to_storage=filename_ecotaxa)
    # Compress folder to prepare upload on Ecotaxa
    shutil.make_archive(str(save_directory), 'zip', save_directory, base_dir=None)
    # Generate a project on Ecotaxa and upload successive samples
    if 'ecotaxa_lexplore_alga_cytosense_projectid' not in cfg_metadata.keys():
        create_ecotaxa_project(ecotaxa_configuration=configuration,
                               project_config={'clone of id':cfg_metadata['ecotaxa_lexplore_alga_flowcam_micro_projectid'], 'title': 'Lexplore_ALGA_Cytosense',
                                               'instrument': 'CytoSense',
                                               'managers': [cfg_metadata['principal_investigator']]+cfg_metadata['instrument_operator'].split(' / '),
                                               'project_description': 'This dataset includes thumbnails generated by a CytoSense. Acquisitions are done on samples collected on a sub-daily basis at the surface of Lake Geneva as part of the ALGA project (PI: Bastiaan Ibelings)',
                                               'project_sorting_fields':'\r\n'.join(pd.Series(cfg_metadata['ecotaxa_initial_sorting_fields'].split('\r\n'))[('object_'+pd.Series(cfg_metadata['ecotaxa_initial_sorting_fields'].split('\r\n')).str.split('=').str[0]).isin(df_ecotaxa.columns)])},update_configuration=True)

    upload_thumbnails_ecotaxa_project(ecotaxa_configuration=configuration, project_id=int(cfg_metadata['ecotaxa_lexplore_alga_cytosense_projectid']), source_path=str(save_directory)+'.zip')

    # Generate Normalized Biovolume Size Spectra
    df_nbss_sample, df_nbss_boot_sample = nbss_estimates(df=pd.merge(df_properties.query('Sample=="{}"'.format(sample)),df_volume.loc[sample].to_frame().T.assign(volume=lambda x: x.Volume_Fluid_imaged.astype(float))[['volume','Measurement_duration']],how='left',right_index=True,left_on=['Sample']), pixel_size=1, grouping_factor=['Sample']) # Rest pixel size to 1 since pixel units were already converted to metric units
    df_nbss=pd.concat([df_nbss,df_nbss_sample],axis=0).reset_index(drop=True)



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
#Attention, grouping factor should be a string
plot = (ggplot(df_nbss) +
        #stat_summary(data=df_nbss_boot_sample.melt(id_vars=['Group_index','Sample','size_class_mid'],value_vars='NBSS'),mapping=aes(x='size_class_mid', y='value',group='Sample',fill='Sample'),geom='ribbon',alpha=0.1,fun_data="median_hilow",fun_args={'confidence_interval':0.95})+
        #geom_ribbon(mapping=aes(x='size_class_mid', y='NBSS',ymin='np.maximum(0,NBSS-NBSS_std/2)',ymax='NBSS+NBSS_std/2',group='Sample',color='Sample'),alpha=0.1)+
        geom_point(mapping=aes(x='size_class_mid', y='NBSS',group='Group_index',colour='Sample'), alpha=1)+
        labs(x='Equivalent circular diameter ($\mu$m)',y='Normalized Biovolume Size Spectra ($\mu$m$^{3}$ mL$^{-1}$ $\mu$m$^{-3}$)', title='',colour='') +
        scale_y_log10(breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+04)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        scale_x_log10( breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e+00)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e+00)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(), labels=lambda l: [v if ((v / (10 ** np.floor(np.log10(v)))) == 1) else '' for v in l]) +
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)
plot.show()
plot.savefig(fname='{}/figures/Initial_test/cytosense_nbss.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')

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