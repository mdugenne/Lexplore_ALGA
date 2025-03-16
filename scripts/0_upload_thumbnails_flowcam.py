## Objective: This script was written to test a set of image processing steps on Python

# Load modules and functions required for image processing
try:
    from funcs_image_processing import *
except:
    from scripts.funcs_image_processing import *


#Digits recognition - This is obsolete as long as mosaic are saved after sorting images by capture ID
#import pytesseract # Use pip install pytesseract. See documentation at https://tesseract-ocr.github.io/tessdoc/Installation.html
#pytesseract.pytesseract.tesseract_cmd = r'{}\AppData\Local\Programs\Tesseract-OCR\tesseract'.format(str(Path.home()))

matplotlib.use('Qt5Agg')

# Identification of the variables of interest in visualspreadsheet summary tables
#Workflow starts here

# Load metadata files (volume imaged, background pixel intensity, etc.) and save entries into separate tables (metadata and statistics)
metadatafiles=natsorted(list(Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data' / 'Lexplore' / 'acquisitions').expanduser().rglob('Flowcam_*_lexplore*/*_summary.csv')))
metadatafiles=list(filter(lambda element: element.parent.parent.stem not in ['Tests','Blanks'],metadatafiles)) # Discard test and blank acquisitions
df_metadata=pd.concat(map(lambda file:(df:=pd.read_csv(file,sep=r'\t',engine='python',encoding='latin-1',names=['Name','Value']),df:=df.Name.str.split(r'\,',n=1,expand=True).rename(columns={0:'Name',1:'Value'}),df:=df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).dropna().reset_index(drop=True) ,(df:=df.assign(Name=lambda x: x.Name.str.replace('========','').str.replace(' ','_').str.strip('_'),Value=lambda x: x.Value.str.lstrip(' ')).set_index('Name').T.rename(index={'Value':file.parent.name.replace(' ','_')})),df:=df[[col for col in df.columns if col in summary_metadata_columns]])[-1],metadatafiles),axis=0)
df_metadata=df_metadata.assign(Ecotaxa_project=lambda x: x.index.astype(str).str.lower().str.split('_').str[0:2].str.join('_').map({'flowcam_10x':'Lexplore_ALGA_Flowcam_micro','flowcam_2mm':'Lexplore_ALGA_Flowcam_macro'}),Instrument='FlowCam')
# Check the fluid volume imaged calculation based on the number of frames used.
df_metadata.Used.astype(float)/df_metadata.Fluid_Volume_Imaged.str.split(' ').str[0].astype(float)
# This should be constant if the area of acceptable region, the size calibration, and the depth of the flow cell are kept constant
df_particle_statistics=pd.concat(map(lambda file:(df:=pd.read_csv(file,sep=r'\t',engine='python',encoding='latin-1',names=['Name','Value']),df:=df.Name.str.split(r'\,',n=1,expand=True).rename(columns={0:'Name',1:'Value'}),df:=df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).reset_index(drop=True),df:=df[df.query('Name.str.contains(r"\=",case=True)').index[0]:].reset_index(drop=True) ,df_summary_statistics:=pd.DataFrame([(df.query('Name.str.contains(r"\=",case=True)').index).tolist(),((df.query('Name.str.contains(r"\=",case=True)').index)[1:]-1).tolist()+[len(df)]]).T.apply(lambda id:{df.loc[id[0],'Name'].split(' ')[1]:pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),df.loc[id[0]+1:id[1],'Value'].values.tolist()[1].split(','))),index=[file.parent.name]) if len(df.loc[id[0]+1:id[1],'Value'].values.tolist())>1 else pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),[np.nan]*len(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(',')))),index=[file.parent.name.replace(' ','_')])},axis=1,result_type='reduce'),final_df:=list(df_summary_statistics[[id for id,key in df_summary_statistics.items() if 'Particle' in key.keys()][0]].values())[0] if len([id for id,key in df_summary_statistics.items() if 'Particle' in key.keys()]) else pd.DataFrame({},index=[file.parent.name.replace(' ','_')]))[-1],metadatafiles),axis=0)
df_filter_statistics=pd.concat(map(lambda file:(df:=pd.read_csv(file,sep=r'\t',engine='python',encoding='latin-1',names=['Name','Value']),df:=df.Name.str.split(r'\,',n=1,expand=True).rename(columns={0:'Name',1:'Value'}),df:=df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).reset_index(drop=True),df:=df[df.query('Name.str.contains(r"\=",case=True)').index[0]:].reset_index(drop=True) ,df_summary_statistics:=pd.DataFrame([(df.query('Name.str.contains(r"\=",case=True)').index).tolist(),((df.query('Name.str.contains(r"\=",case=True)').index)[1:]-1).tolist()+[len(df)]]).T.apply(lambda id:{df.loc[id[0],'Name'].split(' ')[1]:pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),df.loc[id[0]+1:id[1],'Value'].values.tolist()[1].split(','))),index=[file.parent.name]) if len(df.loc[id[0]+1:id[1],'Value'].values.tolist())>1 else pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),[np.nan]*len(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(',')))),index=[file.parent.name.replace(' ','_')])},axis=1,result_type='reduce'),final_df:=list(df_summary_statistics[[id for id,key in df_summary_statistics.items() if 'Filter' in key.keys()][0]].values())[0] if len([id for id,key in df_summary_statistics.items() if 'Filter' in key.keys()]) else pd.DataFrame({},index=[file.parent.name.replace(' ','_')]))[-1],metadatafiles),axis=0)
df_summary_statistics=pd.concat(map(lambda file:(df:=pd.read_csv(file,sep=r'\t',engine='python',encoding='latin-1',names=['Name','Value']),df:=df.Name.str.split(r'\,',n=1,expand=True).rename(columns={0:'Name',1:'Value'}),df:=df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).reset_index(drop=True),df:=df[df.query('Name.str.contains(r"\=",case=True)').index[0]:].reset_index(drop=True) ,df_summary_statistics:=pd.DataFrame([(df.query('Name.str.contains(r"\=",case=True)').index).tolist(),((df.query('Name.str.contains(r"\=",case=True)').index)[1:]-1).tolist()+[len(df)]]).T.apply(lambda id:{df.loc[id[0],'Name'].split(' ')[1]:pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),df.loc[id[0]+1:id[1],'Value'].values.tolist()[1].split(','))),index=[file.parent.name]) if len(df.loc[id[0]+1:id[1],'Value'].values.tolist())>1 else pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),[np.nan]*len(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(',')))),index=[file.parent.name.replace(' ','_')])},axis=1,result_type='reduce'),final_df:=list(df_summary_statistics[[id for id,key in df_summary_statistics.items() if 'Metadata' in key.keys()][0]].values())[0] if len([id for id,key in df_summary_statistics.items() if 'Metadata' in key.keys()]) else pd.DataFrame({},index=[file.parent.name.replace(' ','_')]))[-1],metadatafiles),axis=0)

df_pixel=df_metadata[['Magnification','Calibration_Factor']]

# Load context file for cropping area
df_cropping=df_context[['AcceptableTop','AcceptableBottom','AcceptableLeft','AcceptableRight']]

## Processing vignettes mosaic

runfiles=natsorted(list(Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data' / 'Lexplore' / 'acquisitions' ).expanduser().glob('Flowcam_*_lexplore*')))#Path(r"R:\Imaging_Flowcam\Flowcam data\Lexplore\acquisitions\Flowcam_10x_lexplore_wasam_20241002_2024-10-09_glut\collage_70.png")#imagefiles[538]
# Search for mosaic files (all starting with ***collage***) in local data storage repository
mosaicfiles=dict(map(lambda file: (file.name,natsorted(list(file.rglob('collage_*.png')))),runfiles))

# Loop through sample to (1) generate thumbnails, (2) ecotaxa tables, (3) upload on Ecotaxa, (4) compute Normalized Biovolume Size Spectrum:
df_properties_all=pd.DataFrame()
df_volume=pd.DataFrame()
df_nbss=pd.DataFrame()
for sample in natsorted(list(set(list(mosaicfiles.keys()))-set(natsorted(list(Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data' / 'Lexplore' / 'ecotaxa' ).expanduser().glob('Flowcam_*_lexplore*'))))))[-40:-9]:
    sample_id=sample
    path_to_data = mosaicfiles[sample][0].parent / str(sample + '.csv')
    path_to_ecotaxa = Path(str(mosaicfiles[sample][0]).replace('acquisitions', 'ecotaxa')).expanduser().parent
    df_properties_sample_merged=pd.DataFrame()
    particle_id = 0
    # Reset flow rate in metadat based on context file
    sample = sample.replace(' ', '_')
    df_metadata.loc[sample,'Flow_Rate']=df_context.loc[df_context.index.str.lower().str.split('_').str[1:3].str.join('_').str.contains('_'.join(sample.lower().split('_')[0:2])),'PumpFlowRate'].values[0]+ ' ml/min' if str(df_metadata.loc[sample,'Flow_Rate'])=='nan' else df_metadata.loc[sample,'Flow_Rate']
    df_volume =pd.concat([df_volume,df_metadata.loc[sample].to_frame().T],axis=0)

    if path_to_data.exists():
        df_properties_sample=pd.read_csv(path_to_data,encoding='latin-1',index_col='Capture ID')
    else:
        df_properties_sample=pd.DataFrame(dict(zip(list(dict_properties_visual_spreadsheet.keys()),[pd.NA]*len((dict_properties_visual_spreadsheet.keys())))),index=[0])
    # Load the background image to performs segmentation and re-compute morphometric properties using the same algorithm as CytoSense
    #cropping_area = df_cropping.loc['acquisitions']
    background = ski.io.imread(path_to_data.parent/'cal_image_000001.tif' ,as_gray=True)#[int(cropping_area[0]):int(cropping_area[1]),int(cropping_area[2]):int(cropping_area[3])]
    ##plt.figure(),plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), cmap='gray'),plt.show()

    # Check for particles to skip (e.g. bubbles).
    # A line should be added manually in the summary export files to inform the particles ID (capture ID) that were manually filtered out
    # e.g. Particle Count, 3300
    #      Skip,[2235:2253]+[2255:2264]+[2266]
    if df_metadata.astype({'Skip': str}).loc[sample, 'Skip'] != 'nan':
        id_to_skip = sum(list(map(lambda id: list( np.arange(int(re.sub('\W+', '', id.split(':')[0])), int(re.sub('\W+', '', id.split(':')[1])) + 1)) if len( id.split(':')) == 2 else [int(re.sub('\W+', '', id))], df_metadata.loc[sample, 'Skip'].split(r']+['))), [])
        # if len(id_to_skip)!=(int(df_metadata.astype({'Particle_Count':float}).loc[sample_id,'Particle_Count'])-int(df_summary_statistics.loc[sample_id,'Count'])):
        # print('\nAttention, Number of particles to skip is different than the difference between total particle count and particle count in the "Metadata statistics table".\nSkipping run {}. Please check the summary file and data file for missing particles'.format(sample_id))
        # continue
    else:
        id_to_skip = []
    id_discarded = id_to_skip

    with tqdm(desc='Generating vignettes for run {}'.format(sample_id), total=len(natsorted(mosaicfiles[sample_id])), bar_format='{desc}{bar}', position=0, leave=True) as bar:
        for file in natsorted([str(file) for file in mosaicfiles[sample_id]]):

            file=Path(file).expanduser()
            percent = np.round(100 * (bar.n / len(mosaicfiles[sample_id])), 1)
            bar.set_description('Generating vignettes for run {} (%s%%)'.format(sample) % percent, refresh=True)

            pixel_size = float(df_pixel.at[sample, 'Calibration_Factor'].strip(' '))  # in microns per pixel


            # Crop the bottom pixels (-30 pixels) of the mosaic, otherwise long particles may coincide with the bottom caption and be discarded
            image,colored_image = ski.io.imread(file,as_gray=True)[slice(None, -30, None)],ski.io.imread(file,as_gray=False)[slice(None, -30, None)]
            ##plt.figure(),plt.imshow(image, cmap='gray'),plt.show()

            # Perform segmentation on the mosaic to detect rectangular thumbnails

            edges = ski.filters.sobel(image)
            mask= sp.ndimage.binary_fill_holes(edges)==False
            ##plt.figure(),plt.imshow(mask, cmap='gray'),plt.show()

            # Identify regions of interest
            labelled = measure.label(mask,background=1)
            ##plt.figure(),plt.imshow(labelled),plt.show()

            # Measure region properties to look for rectangular blobs, a.k.a thumbnails
            df_properties=pd.DataFrame(ski.measure.regionprops_table(label_image=labelled,intensity_image=image,properties=['area_convex','area_bbox','axis_major_length','axis_minor_length','bbox','extent','slice']))
            # Identify and plot rectangular regions (=vignettes), except for the last two corresponding to the period in the bottom sentence displayed on the mosaic ('Property shown : Capture ID')
            # Extent is rounded to the 4th decimals to avoid skipping large bubbles that takes the full mosaic height
            rect_idx=df_properties.query('(area_convex==area_bbox) & (extent.round(4)==1)').index+1#df_properties.query('(area_convex==area_bbox) & (extent==1)').index[:-2]+1
            ##plt.figure(),plt.imshow(np.where(np.in1d(labelled,rect_idx).reshape(labelled.shape),image,0), cmap='gray'),plt.show()
            labelid_idx=((df_properties.query('(area_convex!=area_bbox) & (extent!=1)')).sort_values(['bbox-1','bbox-0']).index)+1#((df_properties.query('(area_convex!=area_bbox) & (extent!=1)')[:-4]).sort_values(['bbox-1','bbox-0']).index)+1 #label are sorted according to the x position
            ##plt.figure(),plt.imshow(np.where(np.in1d(labelled,labelid_idx).reshape(labelled.shape),image,0), cmap='gray'),plt.show()

            for id_of_interest in np.arange(0,len(rect_idx)):
                if len(np.where(np.in1d(labelled,rect_idx[id_of_interest]).reshape(labelled.shape),image,0)[df_properties.at[rect_idx[id_of_interest]-1,'slice']][slice(1, -1, None), slice(1, -1, None)]):
                    particle_id = particle_id + 1
                    ''' Deprecated: From now on, the ID to be skipped are still part of the mosaic files to limit the number of manual steps and save time
                    while (particle_id in id_to_skip) : # If the particle was manually filtered out, skip the current id and proceed to the next one
                        particle_id = particle_id + 1
                    '''


                    vignette_id=np.pad(np.where(np.in1d(labelled,rect_idx[id_of_interest]).reshape(labelled.shape),image,0)[df_properties.at[rect_idx[id_of_interest]-1,'slice']][slice(1, -1, None), slice(1, -1, None)],20,constant_values=0)
                    ##plt.figure(),plt.imshow(vignette_id, cmap='gray'),plt.show()
                    ##plt.figure(),plt.imshow(cv2.cvtColor(colored_image[df_properties.at[rect_idx[id_of_interest]-1,'slice']][slice(1, -1, None), slice(1, -1, None)], cv2.COLOR_BGR2RGB)),plt.show()
                    ''' # Obsolete unless some particles got deleted in visualspreadsheet and not properly identified in the sample summary file
                    capture_id=np.pad(np.where(np.in1d(labelled,labelid_idx[id_of_interest]).reshape(labelled.shape),image,0)[df_properties.at[labelid_idx[id_of_interest]-1,'slice']][slice(1, -1, None), slice(1, -1, None)],20,constant_values=0)
                    
                    fig,axes=plt.subplots(1,1)
                    plt.imshow(capture_id, cmap='gray')
                    axes.set_axis_off()
                    ski.filters.try_all_threshold(capture_id)
                    binary= capture_id < ski.filters.threshold_isodata(capture_id)
                    binary= capture_id < ski.filters.threshold_minimum(capture_id)
                    binary= capture_id < ski.filters.threshold_otsu(capture_id)
                    plt.imshow(binary, cmap='gray')
                    
                    
                    pytesseract.image_to_string(binary,config= r'--psm 10')
                    pytesseract.image_to_string(binary,config= r'digits')
                    pytesseract.image_to_string(capture_id==0,config= r'--psm 10')
                    #particle_id=pytesseract.image_to_string(capture_id<(1-np.quantile(capture_id,0.97)),config= r'--psm 10')
                    plt.imshow(capture_id<(1-np.quantile(capture_id,0.97)), cmap='gray')
                    
                    print(pytesseract.image_to_string(np.where(np.in1d(labelled,labelid_idx[0]).reshape(labelled.shape),image,0),config='--psm 13 outputbase digits'))
                    '''

                    #Use background image to perform de-novo segmentation

                    cropped_top=df_properties_sample.loc[particle_id,'Capture Y']
                    cropped_bottom=cropped_top+df_properties_sample.loc[particle_id,'Image Height']
                    cropped_left=df_properties_sample.loc[particle_id,'Capture X']
                    cropped_right = cropped_left+df_properties_sample.loc[particle_id, 'Image Width']
                    background_cropped=background[int(cropped_top):int(cropped_bottom),int(cropped_left):int(cropped_right)]
                    ##plt.figure(),plt.imshow(cv2.cvtColor(background_cropped, cv2.COLOR_BGR2RGB)),plt.show()

                    image_cropped = image[df_properties.at[rect_idx[id_of_interest]-1,'slice']][slice(1, -1, None), slice(1, -1, None)]
                    colored_image_cropped= colored_image[df_properties.at[rect_idx[id_of_interest]-1,'slice']][slice(1, -1, None), slice(1, -1, None)]
                    ##plt.figure(),plt.imshow(colored_image_cropped),plt.show()
                    if background_cropped.shape!=image_cropped.shape: # Background might need to be reshaped since the thumbnails appearing at the bottom of the mosaic are cropped off 30 pixels to avoid overlap with the text
                        cropped_x=-1 * (background_cropped.shape[0] - image_cropped.shape[0]) if  (background_cropped.shape[0] != image_cropped.shape[0]) else None
                        cropped_y = -1 * (background_cropped.shape[1] - image_cropped.shape[1]) if (background_cropped.shape[1] != image_cropped.shape[1]) else None
                        background_cropped = background_cropped[ slice(None,cropped_x , None),slice(None,cropped_y , None)]

                    diff_image = compare_images(image_cropped,background_cropped, method='diff')
                    ##plt.figure(),plt.imshow(diff_image),plt.show()
                    edges = ski.filters.sobel(diff_image)
                    markers = np.zeros_like(diff_image)
                    threshold=df_context.loc[df_context.index.str.replace('context_','').str.contains('_'.join(sample.lower().split('_')[0:3])),['ThresholdDark', 'ThresholdLight']]
                    markers[diff_image >(threshold.astype({'ThresholdDark': float, 'ThresholdLight': float})[ ['ThresholdDark', 'ThresholdLight']].min(axis=1).values / 255)] = 1
                    #markers[diff_image > (1/threshold.astype({'ThresholdDark':float, 'ThresholdLight':float})[['ThresholdDark', 'ThresholdLight']].max(axis=1).values)] = 1
                    #markers[edges > np.quantile(edges, 0.85)] = 1
                    markers[(sp.ndimage.binary_closing(edges) == False)] = 0
                    ##plt.figure(), plt.imshow(markers, cmap='gray'), plt.show()
                    fill_image = ski.morphology.erosion(ski.morphology.erosion(ski.morphology.closing(ski.morphology.dilation(markers,mode='constant'))))
                    distance_neighbor=df_context.loc[df_context.index.str.replace('context_','').str.contains('_'.join(sample.lower().split('_')[0:3])),['DistanceToNeighbor']]
                    fill_image = ski.morphology.remove_small_holes(ski.morphology.closing(fill_image, ski.morphology.square(np.max([1,int(np.floor(distance_neighbor.DistanceToNeighbor.astype(float).values[0]/pixel_size/4))]))).astype(bool)) # int(np.floor(int(df_context.DistanceToNeighbor.astype(float).values[0]) / pixel_size))
                    ##plt.figure(), plt.imshow(fill_image, cmap='gray'), plt.show()
                    label_objects, nb_labels = sp.ndimage.label(fill_image)
                    min_esd=df_context.loc[df_context.index.str.replace('context_','').str.contains('_'.join(sample.lower().split('_')[0:3])),['MinESD']]
                    label_objects = ski.morphology.remove_small_objects(label_objects, min_size=float( min_esd.MinESD.values[0]) / pixel_size)
                    label_objects, nb_labels = sp.ndimage.label(label_objects) # Needs to re-assign the label after discarding small objects
                    # plt.figure(),plt.imshow(label_objects),plt.show()
                    # plt.figure(),plt.imshow(label_objects.astype(bool).astype(int)),plt.show()

                    # Segmentation of the background if flow cell was dirty
                    edges = ski.filters.sobel(background_cropped)
                    markers = np.zeros_like(background_cropped)
                    markers[edges > np.quantile(edges, 0.99)] = 1
                    markers[edges > (threshold.astype({'ThresholdDark':float, 'ThresholdLight':float})[['ThresholdDark', 'ThresholdLight']].min(axis=1).values/255)] = 1
                    markers[(sp.ndimage.binary_closing(edges) == False)] = 0

                    # plt.figure(),plt.imshow(markers, cmap='gray'),plt.show()
                    fill_image = sp.ndimage.binary_fill_holes(markers)
                    fill_image = ski.morphology.erosion(ski.morphology.erosion(ski.morphology.closing(ski.morphology.dilation(markers,mode='constant'))))
                    fill_image = ski.morphology.remove_small_holes(ski.morphology.closing(fill_image, ski.morphology.square(np.max([1,int(np.floor(distance_neighbor.DistanceToNeighbor.astype(float).values[0]/pixel_size/4))]))).astype(bool)) # int(np.floor(int(df_context.DistanceToNeighbor.astype(float).values[0]) / pixel_size))

                    # plt.figure(),plt.imshow(fill_image),plt.show()
                    label_objects_background, nb_labels_background = sp.ndimage.label(fill_image)
                    label_objects_background = ski.morphology.remove_small_objects(label_objects_background, min_size=float( min_esd.MinESD.values[0]) / pixel_size)
                    label_objects_background, nb_labels_background = sp.ndimage.label(label_objects_background)
                    ## plt.figure(),plt.imshow(label_objects_background.astype(bool).astype(int)),plt.show()
                    ## plt.figure(), plt.imshow( label_objects_background.astype(bool).astype(int)-(label_objects.astype(bool).astype(int))),plt.show()
                    label_diff=label_objects.copy()
                    #label_diff[(label_objects.astype(bool).astype(int)==1)&(label_objects_background.astype(bool).astype(int)-(label_objects.astype(bool).astype(int))!=-1)]=0
                    label_diff[(label_objects.astype(bool).astype(int) == 1) & ( label_objects_background.astype(bool).astype(int) - (label_objects.astype(bool).astype(int)) != 0)] = 0

                    # plt.figure(),plt.imshow(label_diff),plt.show()

                    # Discard objects with same bounding box plus/minus neighbor pixels
                    ski.measure.regionprops_table(label_image=ski.morphology.dilation(label_objects,shift_x=np.max([1,int(distance_neighbor.DistanceToNeighbor.astype(float).values[0])]),shift_y=np.max([1,int(distance_neighbor.DistanceToNeighbor.astype(float).values[0])])), properties=['slice'])
                    ski.measure.regionprops_table(label_image=ski.morphology.dilation(label_diff, shift_x=np.max([1,int(distance_neighbor.DistanceToNeighbor.astype(float).values[0])]), shift_y=np.max([1,int( distance_neighbor.DistanceToNeighbor.astype(float).values[0])])), properties=['slice'])
                    if any(list(map(lambda object:ski.measure.intersection_coeff(np.isin(label_objects,object).astype(bool).astype(int), label_diff.astype(bool).astype(int)),np.arange(1,nb_labels+1)))) > 0.80: #<
                        id_to_discard=list(np.where(list(map(lambda object:ski.measure.intersection_coeff(np.isin(label_objects,object).astype(bool).astype(int), label_diff.astype(bool).astype(int))>0.80,np.arange(1,nb_labels+1))))[0]+1)#pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects, properties=['slice'])).index[(pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects,properties=['slice'])).slice.isin(pd.DataFrame( ski.measure.regionprops_table(label_image=label_diff, properties=['slice'])).slice.unique()))==True]+1
                    elif (len(pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects,properties=['perimeter'])).query('perimeter<{}'.format(0.05*ski.measure.regionprops_table(label_image=image_cropped.astype(bool).astype(int),properties=['perimeter'])['perimeter'][0])))) | all(pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects,properties=['perimeter'])).values==ski.measure.regionprops_table(label_image=image_cropped.astype(bool).astype(int),properties=['perimeter'])['perimeter'][0]): # Adding a condition for object representing a small fraction (5 percent) of the total frame or the entire frame
                        id_to_discard=list(pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects,properties=['perimeter'])).query('perimeter<{}'.format(0.05*ski.measure.regionprops_table(label_image=image_cropped.astype(bool).astype(int),properties=['perimeter'])['perimeter'][0])).index+1)
                        id_to_discard=id_to_discard+list(pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects,properties=['perimeter'])).query('perimeter=={}'.format(ski.measure.regionprops_table(label_image=image_cropped.astype(bool).astype(int),properties=['perimeter'])['perimeter'][0])).index+1)

                    else:
                        id_to_discard=[]
                    if len(id_to_discard):
                        label_objects[np.in1d(label_objects,id_to_discard).reshape(label_objects.shape)]=0
                    # plt.figure(),plt.imshow(label_objects),plt.show()
                    # plt.figure(),plt.imshow(label_objects.astype(bool).astype(int)),plt.show()
                    label_markers, nb_labels = sp.ndimage.label(label_objects.astype(bool).astype(int))
                    labelled_particle = measure.label(label_objects.astype(bool).astype(int))  # measure.label(ski.morphology.remove_small_holes(markers.astype(bool),connectivity=1))
                    ##plt.figure(), plt.imshow(labelled_particle, cmap='gray_r'), plt.show()
                    if nb_labels>0: #Otherwise, particle was entirely in the background and should be discarded
                        largest_object = labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1
                        ##plt.figure(), plt.imshow(largest_object, cmap='gray_r'), plt.show()
                        largest_object = np.isin(labelled_particle, np.arange(0, len((np.bincount(labelled_particle.flat)[ 1:]))) + 1)  # labelled == np.argmax(np.bincount(labelled.flat)[1:])+1

                        # Check if particle ID should be skipped
                        if particle_id not in id_to_skip:
                            # Measure properties and append to existing df_properties_sample_merged

                            df_properties_git = pd.concat([pd.DataFrame({'Capture ID':str(particle_id).rstrip(),'img_file_name': 'thumbnail_{}_{}.jpg'.format(str(sample_id).rstrip().replace(' ','_'), str(particle_id).rstrip()), 'Sample': sample_id.replace(' ','_'), 'nb_particles': nb_labels}, index=[particle_id]), pd.DataFrame( ski.measure.regionprops_table(label_image=largest_object.astype(int), intensity_image=colored_image_cropped, properties=['area', 'area_bbox', 'area_convex', 'area_filled','axis_major_length', 'axis_minor_length', 'axis_major_length', 'bbox', 'centroid_local','centroid_weighted_local', 'eccentricity','equivalent_diameter_area', 'extent', 'image_intensity','inertia_tensor', 'inertia_tensor_eigvals','intensity_mean', 'intensity_max', 'intensity_min','intensity_std', 'moments', 'moments_central', 'num_pixels', 'orientation', 'perimeter', 'slice'],spacing=pixel_size), index=[particle_id])], axis=1) if nb_labels > 0 else pd.DataFrame({'img_file_name': 'thumbnail_{}_{}.png'.format(str(sample_id).rstrip(), str(particle_id).rstrip()), 'Sample': sample_id, 'nb_particles': nb_labels}, index=[particle_id])
                            df_properties_sample_merged = pd.concat([df_properties_sample_merged, df_properties_git],axis=0).reset_index(drop=True)

                            # Compare ellipsoid area to convex area to identify bubbles
                            df_properties_git=df_properties_git.assign(circular_check=lambda x: 4*np.pi*x.area/(x.perimeter)**2,ellipsoidal_check=lambda x: x.area_convex/(np.pi*(x.axis_major_length/2)*(x.axis_minor_length/2)),color_check=lambda x:all(np.mean(np.mean(x.at[particle_id,'image_intensity'], axis=1), axis=0)<50) & (all(np.diff(np.mean(np.mean(x.at[particle_id,'image_intensity'], axis=1), axis=0))<10)))


                            # Split the mosaic according to the slices of rectangular regions to generate thumbnail and save
                            contour = ski.morphology.dilation(largest_object.astype(int), footprint=ski.morphology.square(3), out=None, shift_x=False, shift_y=False)
                            contour -= largest_object.astype(int)

                            scale_value = 300 if df_metadata.loc[sample,'Ecotaxa_project'].lower()=='lexplore_alga_flowcam_macro' else 50 # size of the scale bar in microns
                            padding = int((np.ceil(scale_value / pixel_size) + 10) / 2)
                            px_res_matplotlib = 1 /300# plt.rcParams['figure.dpi']


                            #plt.imshow(np.lib.pad(contour, ((25, 25), (padding, padding)), constant_values=0), cmap='gray_r')
                            padded_image=np.lib.pad(colored_image[df_properties.at[rect_idx[id_of_interest] - 1, 'slice']][slice(1, -1, None), slice(1, -1, None)], ((25, 25), (padding, padding), (0, 0)), 'constant', constant_values=np.nanmean(colored_image[df_properties.at[rect_idx[id_of_interest] - 1, 'slice']][slice(1, -1, None), slice(1, -1, None)][np.invert(largest_object)]))
                            fig, axes = plt.subplots(1, 1, frameon=False) #figsize=tuple(np.array(padded_image.shape)[0:2][::-1]*40/300),dpi=300
                            plt.imshow(padded_image, cmap='gray')

                            axes.set_axis_off()
                            scalebar = AnchoredSizeBar(transform=axes.transData, size=50 / pixel_size,
                                                       label='50 $\mu$m', loc='lower center',
                                                       pad=0.1,
                                                       color='black',
                                                       frameon=False,
                                                       fontproperties=fontprops)
                            axes.add_artist(scalebar)
                            if df_metadata.loc[sample,'Ecotaxa_project'].lower()=='lexplore_alga_flowcam_macro':
                                scalebar = AnchoredSizeBar(transform=axes.transData, size=300 / pixel_size,
                                                           label='300 $\mu$m', loc='upper center',
                                                           pad=0.1,
                                                           color='black',
                                                           frameon=False,
                                                           fontproperties=fontprops)

                                axes.add_artist(scalebar)
                            # axes.set_title('Particle ID: {}'.format(str(particle_id)))
                            save_directory = Path(str(file.parent).replace('acquisitions', 'ecotaxa')).expanduser().parent / sample
                            save_directory.mkdir(parents=True, exist_ok=True)
                            fig.savefig(fname=str(save_directory / 'thumbnail_{}_{}.jpg'.format(str(sample).rstrip(), str(particle_id).rstrip())),transparent=False,  bbox_inches="tight",pad_inches=0, dpi=300)
                            plt.close('all')
                            #Image Saving raw thumbnail for CNN feature extraction
                            thumbnail=Image.fromarray(colored_image[df_properties.at[rect_idx[id_of_interest] - 1, 'slice']][slice(1, -1, None), slice(1, -1, None)])
                            save_raw_directory = Path(str(file.parent).replace('acquisitions', 'cnn' )).expanduser().parent /'input' / sample
                            save_raw_directory.mkdir(parents=True, exist_ok=True)
                            thumbnail.save(str(save_raw_directory / 'thumbnail_{}_{}.jpg'.format(str(sample).rstrip(), str(particle_id).rstrip())))
                    else:
                        id_discarded=id_discarded+[particle_id]

            bar.update(n=1)
    # Merge the de-novo properties datatable with FlowCam data, re-format for EcoTaxa and save
    df_properties_merged=pd.merge(df_properties_sample_merged,df_properties_sample.drop(columns=['Name']).rename(columns=dict(zip(df_properties_sample.columns,'Visualspreadsheet '+df_properties_sample.columns))).reset_index().astype({'Capture ID':str}).assign(Sample=sample),how='left',on=['Sample','Capture ID'])
    df_properties_all=pd.concat([df_properties_all,  df_properties_merged],axis=0).reset_index(drop=True)

    #Check for bubbles which are too numerous with FlowCam macro to track
    df_properties_merged= df_properties_merged.assign(circular_check=lambda x: 4*np.pi*x.area/(x.perimeter)**2,ellipsoidal_check=lambda x: x.area_convex/(np.pi*(x.axis_major_length/2)*(np.maximum(x.axis_minor_length,1)/2)))
    df_properties_merged['color_check']=df_properties_merged.image_intensity.apply(lambda x: all(np.mean(np.mean(x, axis=1), axis=0)<55) & (all(np.diff(np.mean(np.mean(x, axis=1), axis=0))<10)))
    df_properties_merged['spherical_check'] = df_properties_merged.axis_major_length/np.maximum(df_properties_merged.axis_minor_length,1)
    df_pca=pd.concat([pd.concat([df_properties_merged[['circular_check','ellipsoidal_check','spherical_check']],df_properties_merged.image_intensity.apply(lambda x: pd.Series(np.mean(np.mean(x, axis=1), axis=0)))],axis=1),df_properties_merged.image_intensity.apply(lambda x: pd.Series(np.diff(np.mean(np.mean(x, axis=1), axis=0))))],axis=1)
    df_discarded= df_properties_merged.set_index('Capture ID')[((df_properties_merged.set_index('Capture ID').spherical_check <= 1.1)  & (df_properties_merged.set_index('Capture ID').circular_check < 0.9)) & ((df_properties_merged.set_index('Capture ID').ellipsoidal_check > 0.8)  & (df_properties_merged.set_index('Capture ID').ellipsoidal_check<1.1))  & (df_properties_merged.set_index('Capture ID').color_check) & (df_properties_merged.set_index('Capture ID').circular_check > 0.5)] #
    if len(df_discarded): # Looking for additional particles to discard (e.g. background particles that move with the flow of the FlowCam Macro, bubbles)

        # Use features extracted to identify duplicated particles resulting from a flow random orientation
        background_cropped = background[int(df_cropping.loc[df_cropping.index.astype(str).str.contains('_'.join(sample.split('_')[0:2]).lower()),'AcceptableTop']):int(df_cropping.loc[df_cropping.index.astype(str).str.contains('_'.join(sample.split('_')[0:2]).lower()),'AcceptableBottom']), int(df_cropping.loc[df_cropping.index.astype(str).str.contains('_'.join(sample.split('_')[0:2]).lower()),'AcceptableLeft']):int(df_cropping.loc[df_cropping.index.astype(str).str.contains('_'.join(sample.split('_')[0:2]).lower()),'AcceptableRight'])]
        ##plt.figure(),plt.imshow(cv2.cvtColor(background_cropped, cv2.COLOR_BGR2RGB)),plt.show()

        edges = ski.filters.sobel(background_cropped)
        markers = np.zeros_like(background_cropped)
        markers[edges > np.quantile(edges, 0.99)] = 1
        markers[edges > (threshold.astype({'ThresholdDark': float, 'ThresholdLight': float})[['ThresholdDark', 'ThresholdLight']].min(axis=1).values / 255)] = 1
        markers[(sp.ndimage.binary_closing(edges) == False)] = 0
        ##plt.figure(), plt.imshow(markers, cmap='gray'), plt.show()
        fill_image = sp.ndimage.binary_fill_holes(markers)
        fill_image = ski.morphology.erosion(ski.morphology.erosion(ski.morphology.closing(ski.morphology.dilation(markers, mode='constant'))))
        fill_image = ski.morphology.remove_small_holes(ski.morphology.closing(fill_image, ski.morphology.square(np.max([1, int(np.floor(distance_neighbor.DistanceToNeighbor.astype(float).values[0] / pixel_size / 4))]))).astype(bool))
        ## plt.figure(),plt.imshow(fill_image),plt.show()
        label_objects_background, nb_labels_background = sp.ndimage.label(fill_image)
        label_objects_background = ski.morphology.remove_small_objects(label_objects_background, min_size=float(min_esd.MinESD.values[0]) / pixel_size)
        label_objects_background, nb_labels_background = sp.ndimage.label(label_objects_background)
        ## plt.figure(),plt.imshow(label_objects_background),plt.show()
        df_background_properties=pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects_background,properties=['image_filled','area', 'area_bbox', 'area_convex', 'area_filled','axis_major_length', 'axis_minor_length', 'axis_major_length','bbox', 'centroid_local', 'eccentricity', 'equivalent_diameter_area', 'extent', 'inertia_tensor', 'inertia_tensor_eigvals', 'moments', 'moments_central', 'num_pixels','orientation', 'perimeter', 'slice'], spacing=pixel_size))
        properties_to_thumbnails(image=background_cropped, df_properties=df_background_properties, save_directory=save_raw_directory/ 'Background')

        # t-SNE clustering of raw image features
        df_features = image_feature_dataset(image_path=save_raw_directory,filter=df_properties_merged.img_file_name, model_name='resnet18', layer_name='avgpool')
        model_cluster = TSNE(n_components=3)
        df_projection = pd.DataFrame(model_cluster.fit_transform(pd.DataFrame(normalize(df_features[df_features.columns[np.arange(2,df_features.shape[1])]]))))
        df_projection['image_url']=df_features.path
        # clustering outliers
        neighbors = NearestNeighbors(n_neighbors=12)
        neighbors_fit = neighbors.fit(df_projection[[1,2]].to_numpy())
        distances, indices = neighbors_fit.kneighbors(df_projection[[1,2]].to_numpy())

        dbscan = DBSCAN(eps=0.97, min_samples=12,leaf_size=2)
        cluster=dbscan.fit(df_projection[[1,2]].to_numpy())
        lof = LocalOutlierFactor(n_neighbors=10)
        lof.fit_predict(df_projection[[1,2]].to_numpy())
        df_projection['score']=lof.negative_outlier_factor_
        df_projection['cluster']=cluster.labels_
        ##scatter_2d_images(df_projection.assign(color=lambda x:x.cluster.astype(str).isin(['0']),size=lambda x:np.where(x.score<-1.5,30,4)).rename(columns={2:'x',1:'y','image_url':'images'})).run_server( use_reloader=False)
        if any(df_projection.score<-2):
            df_discarded = pd.merge(df_properties_merged, df_projection.assign(img_file_name=df_projection.image_url.apply(lambda path: Path(path).stem+'.jpg')),how='left',on='img_file_name')
            df_outliers=df_discarded.query('cluster!=0')
            id_discarded = id_discarded + natsorted(df_outliers['Capture ID'].astype(int).tolist()+list(filter(None,list(map(lambda id: (int(id)+1) if ((np.min(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)+1)]),'Visualspreadsheet Capture X'])-int(distance_neighbor.DistanceToNeighbor.astype(float).values[0])<=np.max(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)+1)]),'Visualspreadsheet Capture X'])<np.min(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)+1)]),'Visualspreadsheet Capture X'])+int(distance_neighbor.DistanceToNeighbor.astype(float).values[0])) &  (np.min(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)+1)]),'Visualspreadsheet Capture Y'])-int(distance_neighbor.DistanceToNeighbor.astype(float).values[0])<=np.max(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)+1)]),'Visualspreadsheet Capture Y'])<np.min(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)+1)]),'Visualspreadsheet Capture Y'])+int(distance_neighbor.DistanceToNeighbor.astype(float).values[0]))) else (int(id)-1) if ((np.min(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)-1)]),'Visualspreadsheet Capture X'])-int(distance_neighbor.DistanceToNeighbor.astype(float).values[0])<=np.max(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)-1)]),'Visualspreadsheet Capture X'])<np.min(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)-1)]),'Visualspreadsheet Capture X'])+int(distance_neighbor.DistanceToNeighbor.astype(float).values[0])) &  (np.min(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)-1)]),'Visualspreadsheet Capture Y'])-int(distance_neighbor.DistanceToNeighbor.astype(float).values[0])<=np.max(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)-1)]),'Visualspreadsheet Capture Y'])<np.min(df_discarded.loc[df_discarded['Capture ID'].isin([id,str(int(id)-1)]),'Visualspreadsheet Capture Y'])+int(distance_neighbor.DistanceToNeighbor.astype(float).values[0]))) else None,df_outliers['Capture ID'])))))
            #df_test=df_properties_merged[df_properties_merged['Capture ID'].isin(list(map(str,natsorted(id_discarded))))]

    id_discarded = natsorted(np.unique(id_discarded + list(compress(list(np.arange(1, particle_id + 1)),pd.Series(np.arange(1, particle_id + 1)).astype(str).isin(df_properties_merged['Capture ID'].unique()) == False))))

    if len(id_discarded):
        # Add skipped id to summary file for reproducibility
        with Path(str(path_to_data).replace('.csv','_summary.csv')).open(mode='r') as f:
            lines = f.readlines()
        if any(pd.Series(lines).str.startswith('Skip')):
            with Path(str(path_to_data).replace('.csv', '_summary.csv')).open(mode='w') as f:
                for line in pd.Series(lines)[pd.Series(lines).str.startswith('Skip')==False].to_list():
                    f.write(line)

        with Path(str(path_to_data).replace('.csv','_summary.csv')).open(mode='a') as file:
            file.write(r'{}'.format('Skip,'+eval(repr('+'.join(['[%d]' % s if s == e else '[%d:%d]' % (s, e) for (s, e) in format_id_to_skip(natsorted(pd.Series(id_discarded).astype(int).tolist()))])))))
        # Erase outlier images and remove id from properties table to discard in ecotaxa table
        for image in  '{}thumbnail_{}_'.format(str(save_directory )+os.sep,str(sample).rstrip()) + pd.Series( list(map(str,id_discarded)) )+'.jpg':
            Path(image).unlink(missing_ok=True)
        df_properties_merged=df_properties_merged[df_properties_merged['Capture ID'].astype(str).isin(list(map(str,id_discarded)))==False].reset_index(drop=True)

    filename_ecotaxa = str(save_directory.parent / save_directory.stem.rstrip().replace(' ','_') /'ecotaxa_table_{}.tsv'.format(str(sample).rstrip()))
    df_ecotaxa = generate_ecotaxa_table(df=pd.merge( df_properties_merged,df_volume.loc[sample].to_frame().T.assign(acq_acquisition_date=lambda x: pd.to_datetime(x.Start_Time,format='%Y-%m-%d %H:%M:%S').dt.floor('1d').dt.strftime('%Y%m%d'),Sample_Volume_Processed=lambda x: x.Sample_Volume_Processed.str.replace(' ml','').astype(float),Sample_Volume_Aspirated=lambda x: x.Sample_Volume_Aspirated.str.replace(' ml','').astype(float),Fluid_Volume_Imaged=lambda x: x.Fluid_Volume_Imaged.str.replace(' ml','').astype(float),Flow_Rate=lambda x:x.Flow_Rate.str.replace(' ml/min','').astype(float)).rename(columns={'Sample_Volume_Processed':'sample_volume_analyzed_ml','Sample_Volume_Aspirated':'sample_volume_pumped_ml','Fluid_Volume_Imaged':'sample_volume_fluid_imaged_ml','Sampling_Time':'sample_duration_sec','Flow_Rate':'sample_flow_rate'})[['acq_acquisition_date','sample_volume_analyzed_ml','sample_volume_pumped_ml','sample_volume_fluid_imaged_ml','sample_duration_sec','sample_flow_rate']],how='left',right_index=True,left_on=['Sample']), instrument='FlowCam', path_to_storage=filename_ecotaxa)

    # Compress folder to prepare upload on Ecotaxa
    shutil.make_archive(str(save_directory),'zip',save_directory,base_dir=None)

    # Generate a project on Ecotaxa and upload successive samples
    if 'ecotaxa_'+df_metadata.loc[sample,'Ecotaxa_project'].lower()+'_projectid' not in cfg_metadata.keys():
        create_ecotaxa_project(ecotaxa_configuration=configuration,
                               project_config={'clone of id':cfg_metadata['ecotaxa_lexplore_alga_flowcam_micro_projectid'], 'title': 'Lexplore_ALGA_Flowcam_{}'.format(df_metadata.loc[sample,'Ecotaxa_project'].split('_')[-1]),
                                               'instrument': 'FlowCam',
                                               'managers': [cfg_metadata['principal_investigator']]+cfg_metadata['instrument_operator'].split(' / '),
                                               'project_description': 'This dataset includes thumbnails generated by a FlowCam {}. Acquisitions are done on samples collected on a daily basis at the surface of Lake Geneva as part of the ALGA project (PI: Bastiaan Ibelings)'.format(df_metadata.loc[sample,'Ecotaxa_project'].split('_')[-1]),'project_sorting_fields':'\r\n'.join(pd.Series(cfg_metadata['ecotaxa_initial_sorting_fields'].split('\r\n'))[('object_'+pd.Series(cfg_metadata['ecotaxa_initial_sorting_fields'].split('\r\n')).str.split('=').str[0]).isin(df_ecotaxa.columns)])},update_configuration=True)

    upload_thumbnails_ecotaxa_project(ecotaxa_configuration=configuration, project_id=int(cfg_metadata['ecotaxa_'+df_metadata.loc[sample,'Ecotaxa_project'].lower()+'_projectid']), source_path=str(save_directory)+'.zip')

    # Generate Normalized Biovolume Size Spectra
    df_nbss_sample, df_nbss_boot_sample = nbss_estimates(df=pd.merge( df_properties_merged,df_volume.loc[sample].to_frame().T.assign(volume=lambda x: x.Fluid_Volume_Imaged.str.replace(' ml','').astype(float))[['volume']],how='left',right_index=True,left_on=['Sample']), pixel_size=1, grouping_factor=['Sample']) # Set pixel size to 1 since pixel units were already converted to metric units
    df_nbss=pd.concat([df_nbss,df_nbss_sample],axis=0).reset_index(drop=True)

# Plot the Normalized Biovolume Size Spectra
#Attention, grouping factor should be a string
#df_nbss=pd.concat(map(lambda path_ecotaxa:(nbss_estimates(df=pd.read_csv(path_ecotaxa,sep='\t').drop(index=[0]).astype({'object_area':float,'sample_volume_fluid_imaged_ml':float}).rename(columns={'object_area':'area','sample_volume_fluid_imaged_ml':'volume'}), pixel_size=1, grouping_factor=['sample_id'])[0]).assign(instrument=lambda x:np.where(x.sample_id.str.contains('Flowcam_2mm'),'FlowCam Macro','FlowCam Micro')),natsorted(list(save_directory.parent.rglob('ecotaxa_table_*'))))).reset_index(drop=True).rename(columns={'sample_id':'Sample'})
#df_nbss=pd.concat(map(lambda path_ecotaxa:(nbss_estimates(df=pd.read_csv(path_ecotaxa,sep='\t').drop(index=[0]).astype({'object_area':float,'sample_volume_fluid_imaged_ml':float}).rename(columns={'object_area':'area','sample_volume_fluid_imaged_ml':'volume'}), pixel_size=1, grouping_factor=['sample_id'])[0]).assign(instrument="CytoSense"),natsorted(list(Path(path_to_network / 'lexplore' / 'LEXPLORE' / 'ecotaxa' ).rglob('ecotaxa_table_*'))))).reset_index(drop=True).rename(columns={'sample_id':'Sample'})
plot = (ggplot(df_nbss) +
        #geom_point(mapping=aes(x='(1/6)*np.pi*(size_class_mid**3)', y='NBSS'), alpha=1) +  #
        #stat_summary(data=df_nbss_boot_sample.melt(id_vars=['Group_index','Sample','size_class_mid'],value_vars='NBSS'),mapping=aes(x='size_class_mid', y='value',group='Sample',fill='Sample'),geom='ribbon',alpha=0.1,fun_data="median_hilow",fun_args={'confidence_interval':0.95})+
        #geom_ribbon(mapping=aes(x='size_class_mid', y='NBSS',ymin='np.maximum(0,NBSS-NBSS_std/2)',ymax='NBSS+NBSS_std/2',group='Sample',color='Sample'),alpha=0.1)+
        geom_point(mapping=aes(x='size_class_mid', y='NBSS',group='Group_index',colour='instrument'), alpha=1)+
        scale_colour_manual(values={'FlowCam Micro':'#{:02x}{:02x}{:02x}'.format(255,212,42),'FlowCam Macro':'#{:02x}{:02x}{:02x}'.format(152,95,95),'CytoSense':'#{:02x}{:02x}{:02x}'.format(76,95,95)})+
        labs(x='Equivalent circular diameter ($\mu$m)',y='Normalized Biovolume Size Spectra ($\mu$m$^{3}$ mL$^{-1}$ $\mu$m$^{-3}$)', title='',colour='') +
        scale_y_log10(breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+04)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        scale_x_log10( breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e+00)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e+00)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(), labels=lambda l: [v if ((v / (10 ** np.floor(np.log10(v)))) == 1) else '' for v in l]) +
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)
plot.show()
plot.savefig(fname='{}/figures/Initial_test/nbss.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')
# Plot the sizes comparison
plot = (ggplot(df_properties_all.assign(instrument=lambda x: x.Sample.str.split('_').str[0:2].str.join('_'))) +
        geom_point(mapping=aes(x='equivalent_diameter_area', y='Visualspreadsheet Diameter (ABD)',fill='instrument'),size=0.1, alpha=1) +  #
        geom_abline(slope=1,intercept=0)+
        labs(x='Equivalent circular diameter from de-novo segmentation ($\mu$m)',y='Equivalent circular diameter from vp segmentation ($\mu$m)', title='',colour='') +
        scale_y_log10(limits=[1,10000],breaks=np.multiply(10 ** np.arange(np.floor(np.log10(1e+00)), np.ceil(np.log10(1e+04)), step=1).reshape(int((np.ceil(np.log10(1e+00)) - np.floor(np.log10(1e+04)))), 1),np.arange(1, 10, step=1).reshape(1, 9)).flatten(),labels=lambda l: [v if ((v / (10 ** np.floor(np.log10(v)))) == 1) else '' for v in l]) +
        scale_x_log10(limits=[1,10000], breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e+00)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e+00)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(), labels=lambda l: [v if ((v / (10 ** np.floor(np.log10(v)))) == 1) else '' for v in l]) +
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)

''' Deprecated : Thumbnails are now generated from mosaic files since saving raw frames during FlowCam acquisitions result in lower imaging efficiency (70 to 20 percent efficiency)
## Processing raw images
# Load raw and calibration images
outputfiles_raw = list(Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data' ).expanduser().rglob('Flowcam_10x_lexplore*/*.tif'))
backgroundfiles,imagefiles=list(filter(lambda x: 'cal_' in x.name, outputfiles_raw)),list(filter(lambda x: 'rawfile_' in x.name, outputfiles_raw))

# Load image file
file=natsorted(imagefiles)[-1]#Path('R:Imaging_Flowcam/Flowcam data/Lexplore/acquisitions/Flowcam_10x_lexplore_wasam_20241002_2024-10-03 07-23-04/rawfile_011039.tif')#imagefiles[538]
sample_id=file.parent.name
cropping_area=df_cropping.loc['acquisitions']
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
plt.figure(),plt.imshow(filled_image, cmap='gray'),plt.show()

label_objects, nb_labels = sp.ndimage.label(filled_image)
labelled = measure.label(filled_image)#measure.label(ski.morphology.remove_small_holes(markers.astype(bool),connectivity=1))
plt.figure(),plt.imshow(labelled, cmap='gray_r'),plt.show()

2*np.sqrt(np.bincount(labelled.flat)*(pixel_size.loc[sample]**2)/np.pi)>float(df_context.MinESD.values[0])
largest_object = labelled == np.argmax(np.bincount(labelled.flat)[1:])+1
plt.figure(),plt.imshow(largest_object, cmap='gray_r'),plt.show()
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
'''