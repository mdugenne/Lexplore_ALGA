## Objective: This script was written to test a set of image processing steps on Python
import pandas as pd
import skimage.morphology


# Load modules and functions required for image processing

try:
    from funcs_image_processing import *

except:
    from scripts.funcs_image_processing import *


matplotlib.use('Qt5Agg')

# Identification of the variables of interest in visualspreadsheet summary tables
# Workflow starts here
path_to_network=Path('F:\Tercier\A repasser') #Reset network path to local storage
# Load metadata files (volume imaged, background pixel intensity, etc.) and save entries into separate tables (metadata and statistics)
metadatafiles = natsorted(list( Path(path_to_network).expanduser().rglob('*_summary.csv')))
df_metadata = pd.concat(map(lambda file: (df := pd.read_csv(file, sep=r'\t', engine='python', encoding='latin-1', names=['Name', 'Value']),df := df.Name.str.split(r'\,', n=1, expand=True).rename(columns={0: 'Name', 1: 'Value'}),df := df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).dropna().reset_index(drop=True), (df := df.assign(Name=lambda x: x.Name.str.replace('========', '').str.replace(' ', '_').str.strip('_'),Value=lambda x: x.Value.str.lstrip(' ')).set_index('Name').T.rename( index={'Value': file.parent.name.replace(' ', '_')})),df := df[[col for col in df.columns if col in summary_metadata_columns]])[-1], metadatafiles), axis=0)
df_metadata = df_metadata.assign(Ecotaxa_project=None, Instrument='FlowCam')
# Check the fluid volume imaged calculation based on the number of frames used.
df_metadata.Used.astype(float) / df_metadata.Fluid_Volume_Imaged.str.split(' ').str[0].astype(float)
# This should be constant if the area of acceptable region, the size calibration, and the depth of the flow cell are kept constant
df_particle_statistics = pd.concat(map(lambda file: (df := pd.read_csv(file, sep=r'\t', engine='python', encoding='latin-1', names=['Name', 'Value']),df := df.Name.str.split(r'\,', n=1, expand=True).rename(columns={0: 'Name', 1: 'Value'}),df := df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).reset_index(drop=True),df := df[df.query('Name.str.contains(r"\=",case=True)').index[0]:].reset_index(drop=True),df_summary_statistics := pd.DataFrame([(df.query('Name.str.contains(r"\=",case=True)').index).tolist(),((df.query('Name.str.contains(r"\=",case=True)').index)[1:] - 1).tolist() + [len(df)]]).T.apply(lambda id: { df.loc[id[0], 'Name'].split(' ')[1]: pd.DataFrame(dict( zip(df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[0].split(','), df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[1].split(','))), index=[file.parent.name]) if len( df.loc[id[0] + 1:id[1], 'Value'].values.tolist()) > 1 else pd.DataFrame(dict( zip(df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[0].split(','),[np.nan] * len(df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[0].split(',')))), index=[file.parent.name.replace(' ', '_')])}, axis=1, result_type='reduce'), final_df :=list(df_summary_statistics[[id for id, key in df_summary_statistics.items() if 'Particle' in key.keys()][0]].values())[0] if len([id for id, key in df_summary_statistics.items() if 'Particle' in key.keys()]) else pd.DataFrame({}, index=[ file.parent.name.replace(' ', '_')]))[-1], metadatafiles), axis=0)
df_filter_statistics = pd.concat(map(lambda file: (df := pd.read_csv(file, sep=r'\t', engine='python', encoding='latin-1', names=['Name', 'Value']),df := df.Name.str.split(r'\,', n=1, expand=True).rename(columns={0: 'Name', 1: 'Value'}),df := df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).reset_index(drop=True),df := df[df.query('Name.str.contains(r"\=",case=True)').index[0]:].reset_index(drop=True),df_summary_statistics := pd.DataFrame([(df.query('Name.str.contains(r"\=",case=True)').index).tolist(),((df.query('Name.str.contains(r"\=",case=True)').index)[1:] - 1).tolist() + [ len(df)]]).T.apply(lambda id: { df.loc[id[0], 'Name'].split(' ')[1]: pd.DataFrame(dict( zip(df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[0].split(','), df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[1].split(','))), index=[file.parent.name]) if len( df.loc[id[0] + 1:id[1], 'Value'].values.tolist()) > 1 else pd.DataFrame(dict( zip(df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[0].split(','),[np.nan] * len(df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[0].split(',')))), index=[file.parent.name.replace(' ', '_')])}, axis=1, result_type='reduce'), final_df :=list(df_summary_statistics[[id for id, key in df_summary_statistics.items() if 'Filter' in key.keys()][0]].values())[0] if len([id for id, key in df_summary_statistics.items() if 'Filter' in key.keys()]) else pd.DataFrame({}, index=[file.parent.name.replace(' ', '_')]))[-1], metadatafiles), axis=0)
df_summary_statistics = pd.concat(map(lambda file: (df := pd.read_csv(file, sep=r'\t', engine='python', encoding='latin-1', names=['Name', 'Value']),df := df.Name.str.split(r'\,', n=1, expand=True).rename(columns={0: 'Name', 1: 'Value'}),df := df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).reset_index(drop=True),df := df[df.query('Name.str.contains(r"\=",case=True)').index[0]:].reset_index(drop=True),df_summary_statistics := pd.DataFrame([(df.query('Name.str.contains(r"\=",case=True)').index).tolist(), ((df.query('Name.str.contains(r"\=",case=True)').index)[1:] - 1).tolist() + [ len(df)]]).T.apply(lambda id: {df.loc[id[0], 'Name'].split(' ')[1]: pd.DataFrame(dict(zip(df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[0].split(','),df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[1].split(','))), index=[file.parent.name]) if len(df.loc[id[0] + 1:id[1], 'Value'].values.tolist()) > 1 else pd.DataFrame(dict(zip(df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[0].split(','),[np.nan] * len(df.loc[id[0] + 1:id[1], 'Value'].values.tolist()[0].split(',')))), index=[file.parent.name.replace(' ', '_')])}, axis=1, result_type='reduce'), final_df :=list(df_summary_statistics[[id for id, key in df_summary_statistics.items() if 'Metadata' in key.keys()][0]].values())[0] if len([id for id, key in df_summary_statistics.items() if 'Metadata' in key.keys()]) else pd.DataFrame({},index=[ file.parent.name.replace( ' ','_')]))[-1], metadatafiles), axis=0)

df_pixel = df_metadata[['Magnification', 'Calibration_Factor']]

# Load context file for cropping area
df_cropping = df_context[['AcceptableTop', 'AcceptableBottom', 'AcceptableLeft', 'AcceptableRight']]

## Processing vignettes mosaic

runfiles = natsorted(list(Path(path_to_network / 'acquisitions').expanduser().glob('*')))
# Search for mosaic files (all starting with ***collage***) in local data storage repository
mosaicfiles = dict(map(lambda file: (file.name, natsorted(list(file.rglob('image*.png')))), runfiles))
df_context=df_context_flowcam_micro
# Loop through sample to (1) generate thumbnails, (2) ecotaxa tables, (3) upload on Ecotaxa, (4) compute Normalized Biovolume Size Spectrum:
df_properties_all = pd.DataFrame()
df_volume = pd.DataFrame()
df_nbss = pd.DataFrame()
for sample in natsorted(list(mosaicfiles.keys()))[8:]:
    df_excel = pd.read_excel(path_to_network.parent / 'outputs' / 'Flowcam_recap_filtre_taille.xlsx', sheet_name='New')
    sample_id = sample
    path_to_data = mosaicfiles[sample][0].parent / str(sample + '.csv')
    path_to_ecotaxa = Path(str(mosaicfiles[sample][0]).replace('A repasser', 'outputs')).expanduser().parent
    #if path_to_ecotaxa.exists():
    #    continue
    df_properties_sample_merged = pd.DataFrame()
    particle_id = 0
    # Reset flow rate in metadat based on context file
    sample = sample.replace(' ', '_')
    df_metadata.loc[sample, 'Flow_Rate'] = df_context.loc[df_context.index.str.lower().str.split('_').str[1:3].str.join('_').str.contains('_'.join(sample.lower().split('_')[0:2])), 'PumpFlowRate'].values[0] + ' ml/min' if str( df_metadata.loc[sample, 'Flow_Rate']) == 'nan' else df_metadata.loc[sample, 'Flow_Rate']
    df_volume = pd.concat([df_volume, df_metadata.loc[sample].to_frame().T], axis=0)

    if path_to_data.exists():
        df_properties_sample = pd.read_csv(path_to_data, encoding='latin-1', index_col='Capture ID')
    #else:
    #    continue#df_properties_sample = pd.DataFrame(dict(zip(list(dict_properties_visual_spreadsheet.keys()), [pd.NA] * len((dict_properties_visual_spreadsheet.keys())))), index=[0])
    # Load the background image to performs segmentation and re-compute morphometric properties using the same algorithm as CytoSense
    # cropping_area = df_cropping.loc['acquisitions']
    background = ski.io.imread(path_to_data.parent / 'cal_image_000001.tif', as_gray=True)  # [int(cropping_area[0]):int(cropping_area[1]),int(cropping_area[2]):int(cropping_area[3])]
    ##plt.figure(),plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), cmap='gray'),plt.show()
    with tqdm(desc='Generating vignettes for run {}'.format(sample_id), total=len(natsorted(mosaicfiles[sample_id])), bar_format='{desc}{bar}', position=0, leave=True) as bar:
        for file in natsorted([str(file) for file in mosaicfiles[sample_id]]):

            file = Path(file).expanduser()
            percent = np.round(100 * (bar.n / len(mosaicfiles[sample_id])), 1)
            bar.set_description('Generating vignettes for run {} (%s%%)'.format(sample) % percent, refresh=True)

            pixel_size = float(df_pixel.at[sample, 'Calibration_Factor'].strip(' '))  # in microns per pixel

            # Crop the bottom pixels (-30 pixels) of the mosaic, otherwise long particles may coincide with the bottom caption and be discarded
            image, colored_image = ski.io.imread(file, as_gray=True)[slice(None, -30, None)], ski.io.imread(file, as_gray=False)[slice(None, -30, None)]
            ##plt.figure(),plt.imshow(image, cmap='gray'),plt.show()

            # Perform segmentation on the mosaic to detect rectangular thumbnails

            edges = ski.filters.sobel(image)
            mask = sp.ndimage.binary_fill_holes(edges) == False
            ##plt.figure(),plt.imshow(mask, cmap='gray'),plt.show()

            # Identify regions of interest
            labelled = measure.label(mask, background=1)
            ##plt.figure(),plt.imshow(labelled),plt.show()

            # Measure region properties to look for rectangular blobs, a.k.a thumbnails
            df_properties = pd.DataFrame(ski.measure.regionprops_table(label_image=labelled, intensity_image=image, properties=['area_convex', 'area_bbox', 'axis_major_length','axis_minor_length', 'bbox','extent', 'slice']))
            # Identify and plot rectangular regions (=vignettes), except for the last two corresponding to the period in the bottom sentence displayed on the mosaic ('Property shown : Capture ID')
            # Extent is rounded to the 4th decimals to avoid skipping large bubbles that takes the full mosaic height
            rect_idx = df_properties.query('(area_convex==area_bbox) & (extent.round(4)==1)').index + 1  # df_properties.query('(area_convex==area_bbox) & (extent==1)').index[:-2]+1
            ##plt.figure(),plt.imshow(np.where(np.in1d(labelled,rect_idx).reshape(labelled.shape),image,0), cmap='gray'),plt.show()
            labelid_idx = ((df_properties.query('(area_convex!=area_bbox) & (extent!=1)')).sort_values(['bbox-1', 'bbox-0']).index) + 1  # ((df_properties.query('(area_convex!=area_bbox) & (extent!=1)')[:-4]).sort_values(['bbox-1','bbox-0']).index)+1 #label are sorted according to the x position
            ##plt.figure(),plt.imshow(np.where(np.in1d(labelled,labelid_idx).reshape(labelled.shape),image,0), cmap='gray'),plt.show()

            # Check for particles to skip (e.g. bubbles).
            # A line should be added manually in the summary export files to inform the particles ID (capture ID) that were manually filtered out
            # e.g. Particle Count, 3300
            #      Skip,[2235:2253]+[2255:2264]+[2266]
            if df_metadata.astype({'Skip': str}).loc[sample, 'Skip'] != 'nan':
                id_to_skip = sum(list(map(lambda id: list(np.arange(int(re.sub('\W+', '', id.split(':')[0])), int(re.sub('\W+', '', id.split(':')[1])) + 1)) if len( id.split(':')) == 2 else [int(re.sub('\W+', '', id))], df_metadata.loc[sample, 'Skip'].split(r']+['))), [])
                # if len(id_to_skip)!=(int(df_metadata.astype({'Particle_Count':float}).loc[sample_id,'Particle_Count'])-int(df_summary_statistics.loc[sample_id,'Count'])):
                # print('\nAttention, Number of particles to skip is different than the difference between total particle count and particle count in the "Metadata statistics table".\nSkipping run {}. Please check the summary file and data file for missing particles'.format(sample_id))
                # continue
            else:
                id_to_skip = []
            for id_of_interest in np.arange(0, len(rect_idx)):
                if len(np.where(np.in1d(labelled, rect_idx[id_of_interest]).reshape(labelled.shape), image, 0)[df_properties.at[rect_idx[id_of_interest] - 1, 'slice']][ slice(1, -1, None), slice(1, -1, None)]):
                    particle_id = particle_id + 1
                    vignette_id = np.pad(np.where(np.in1d(labelled, rect_idx[id_of_interest]).reshape(labelled.shape), image, 0)[ df_properties.at[rect_idx[id_of_interest] - 1, 'slice']][ slice(1, -1, None), slice(1, -1, None)], 20, constant_values=0)
                    ##plt.figure(),plt.imshow(vignette_id, cmap='gray'),plt.show()
                    ##plt.figure(),plt.imshow(cv2.cvtColor(colored_image[df_properties.at[rect_idx[id_of_interest]-1,'slice']][slice(1, -1, None), slice(1, -1, None)], cv2.COLOR_BGR2RGB)),plt.show()

                    # Use background image to perform de-novo segmentation

                    cropped_top = df_properties_sample.loc[particle_id, 'Capture Y']
                    cropped_bottom = cropped_top + df_properties_sample.loc[particle_id, 'Image Height']
                    cropped_left = df_properties_sample.loc[particle_id, 'Capture X']
                    cropped_right = cropped_left + df_properties_sample.loc[particle_id, 'Image Width']
                    background_cropped = background[int(cropped_top):int(cropped_bottom), int(cropped_left):int(cropped_right)]
                    ##plt.figure(),plt.imshow(cv2.cvtColor(background_cropped, cv2.COLOR_BGR2RGB)),plt.show()

                    image_cropped = image[df_properties.at[rect_idx[id_of_interest] - 1, 'slice']][slice(1, -1, None), slice(1, -1, None)]
                    colored_image_cropped = colored_image[df_properties.at[rect_idx[id_of_interest] - 1, 'slice']][ slice(1, -1, None), slice(1, -1, None)]
                    ##plt.figure(),plt.imshow(colored_image_cropped),plt.show()
                    if background_cropped.shape != image_cropped.shape:  # Background might need to be reshaped since the thumbnails appearing at the bottom of the mosaic are cropped off 30 pixels to avoid overlap with the text
                        cropped_x=-1 * (background_cropped.shape[0] - image_cropped.shape[0]) if  (background_cropped.shape[0] != image_cropped.shape[0]) else None
                        cropped_y = -1 * (background_cropped.shape[1] - image_cropped.shape[1]) if (background_cropped.shape[1] != image_cropped.shape[1]) else None
                        background_cropped = background_cropped[ slice(None,cropped_x , None),slice(None,cropped_y , None)]
                    diff_image = compare_images(image_cropped, background_cropped, method='diff')
                    ##plt.figure(),plt.imshow(diff_image),plt.show()
                    edges = ski.filters.sobel(diff_image)
                    markers = np.zeros_like(diff_image)
                    threshold = df_context[['ThresholdDark', 'ThresholdLight']]
                    markers[diff_image > (threshold.astype({'ThresholdDark': float, 'ThresholdLight': float})[ ['ThresholdDark', 'ThresholdLight']].min(axis=1).values / 255)] = 1
                    # markers[diff_image > (1/threshold.astype({'ThresholdDark':float, 'ThresholdLight':float})[['ThresholdDark', 'ThresholdLight']].max(axis=1).values)] = 1
                    # markers[edges > np.quantile(edges, 0.85)] = 1
                    markers[(sp.ndimage.binary_closing(edges) == False)] = 0
                    ##plt.figure(), plt.imshow(markers, cmap='gray'), plt.show()
                    fill_image = ski.morphology.erosion(ski.morphology.erosion(ski.morphology.closing(ski.morphology.dilation(markers, mode='constant'))))
                    distance_neighbor = df_context[['DistanceToNeighbor']]
                    fill_image = ski.morphology.remove_small_holes(ski.morphology.closing(fill_image,ski.morphology.square(np.max( [1, int(np.floor(distance_neighbor.DistanceToNeighbor.astype(float).values[0] / pixel_size / 4))]))).astype( bool))  # int(np.floor(int(df_context.DistanceToNeighbor.astype(float).values[0]) / pixel_size))
                    ##plt.figure(), plt.imshow(fill_image, cmap='gray'), plt.show()
                    label_objects, nb_labels = sp.ndimage.label(fill_image)
                    min_esd = df_context[['MinESD']]
                    label_objects = ski.morphology.remove_small_objects(label_objects, min_size=float( min_esd.MinESD.values[0]) / pixel_size)
                    label_objects, nb_labels = sp.ndimage.label( label_objects)  # Needs to re-assign the label after discarding small objects
                    # plt.figure(),plt.imshow(label_objects),plt.show()
                    # plt.figure(),plt.imshow(label_objects.astype(bool).astype(int)),plt.show()

                    # Segmentation of the background in case flow cell was dirty
                    edges = ski.filters.sobel(background_cropped)
                    markers = np.zeros_like(background_cropped)
                    markers[edges > np.quantile(edges, 0.99)] = 1
                    markers[edges > (threshold.astype({'ThresholdDark': float, 'ThresholdLight': float})[ ['ThresholdDark', 'ThresholdLight']].min(axis=1).values / 255)] = 1
                    markers[(sp.ndimage.binary_closing(edges) == False)] = 0

                    # plt.figure(),plt.imshow(markers, cmap='gray'),plt.show()
                    fill_image = sp.ndimage.binary_fill_holes(markers)
                    fill_image = ski.morphology.erosion(ski.morphology.erosion(ski.morphology.closing(ski.morphology.dilation(markers, mode='constant'))))
                    fill_image = ski.morphology.remove_small_holes(ski.morphology.closing(fill_image,ski.morphology.square(np.max([1, int(np.floor( distance_neighbor.DistanceToNeighbor.astype(float).values[ 0] / pixel_size / 4))]))).astype( bool))  # int(np.floor(int(df_context.DistanceToNeighbor.astype(float).values[0]) / pixel_size))

                    # plt.figure(),plt.imshow(fill_image),plt.show()
                    label_objects_background, nb_labels_background = sp.ndimage.label(fill_image)
                    label_objects_background = ski.morphology.remove_small_objects(label_objects_background, min_size=float(min_esd.MinESD.values[ 0]) / pixel_size)
                    label_objects_background, nb_labels_background = sp.ndimage.label(label_objects_background)
                    ## plt.figure(),plt.imshow(label_objects_background.astype(bool).astype(int)),plt.show()
                    ## plt.figure(), plt.imshow( label_objects_background.astype(bool).astype(int)-(label_objects.astype(bool).astype(int))),plt.show()
                    label_diff = label_objects.copy()
                    # label_diff[(label_objects.astype(bool).astype(int)==1)&(label_objects_background.astype(bool).astype(int)-(label_objects.astype(bool).astype(int))!=-1)]=0
                    label_diff[(label_objects.astype(bool).astype(int) == 1) & (label_objects_background.astype(bool).astype(int) - ( label_objects.astype(bool).astype(int)) != 0)] = 0

                    # plt.figure(),plt.imshow(label_diff),plt.show()

                    # Discard objects with same bounding box plus/minus neighbor pixels
                    if any(list(map(lambda object: ski.measure.intersection_coeff(np.isin(label_objects, object).astype(bool).astype(int),label_diff.astype(bool).astype(int)), np.arange(1, nb_labels + 1)))) > 0.80:  # <
                        id_to_discard = list(np.where(list(map(lambda object: ski.measure.intersection_coeff(np.isin(label_objects, object).astype(bool).astype(int), label_diff.astype(bool).astype(int)) > 0.80, np.arange(1, nb_labels + 1))))[ 0] + 1)  # pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects, properties=['slice'])).index[(pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects,properties=['slice'])).slice.isin(pd.DataFrame( ski.measure.regionprops_table(label_image=label_diff, properties=['slice'])).slice.unique()))==True]+1
                    else:
                        id_to_discard = []
                    if len(id_to_discard):
                        label_objects[np.in1d(label_objects, id_to_discard).reshape(label_objects.shape)] = 0
                    # plt.figure(),plt.imshow(label_objects),plt.show()
                    # plt.figure(),plt.imshow(label_objects.astype(bool).astype(int)),plt.show()
                    label_markers, nb_labels = sp.ndimage.label(label_objects.astype(bool).astype(int))
                    labelled_particle = measure.label(label_objects.astype(bool).astype( int))  # measure.label(ski.morphology.remove_small_holes(markers.astype(bool),connectivity=1))
                    ##plt.figure(), plt.imshow(labelled_particle, cmap='gray_r'), plt.show()
                    if nb_labels > 0:  # Otherwise, particle was entirely in the background and should be discarded
                        largest_object = labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1
                        ##plt.figure(), plt.imshow(largest_object, cmap='gray_r'), plt.show()
                        largest_object = np.isin(labelled_particle, np.arange(0, len((np.bincount(labelled_particle.flat)[ 1:]))) + 1)  # labelled == np.argmax(np.bincount(labelled.flat)[1:])+1

                        # Check if particle ID should be skipped
                        if particle_id not in id_to_skip:
                            # Measure properties
                            df_properties_git = pd.concat([pd.DataFrame({'Capture ID': str(particle_id).rstrip(), 'img_file_name': 'thumbnail_{}_{}.jpg'.format(str(sample_id).rstrip().replace(' ', '_'), str(particle_id).rstrip()),'Sample': sample_id.replace(' ', '_'),'nb_particles': nb_labels},index=[particle_id]), pd.DataFrame(ski.measure.regionprops_table(label_image=largest_object.astype(int), intensity_image=colored_image_cropped, properties=['area', 'area_bbox', 'area_convex', 'area_filled', 'axis_major_length', 'axis_minor_length', 'axis_major_length', 'bbox', 'centroid_local','centroid_weighted_local', 'eccentricity','equivalent_diameter_area', 'extent','image_intensity', 'inertia_tensor','inertia_tensor_eigvals', 'intensity_mean','intensity_max', 'intensity_min', 'intensity_std', 'moments', 'moments_central', 'num_pixels', 'orientation', 'perimeter','slice'], spacing=pixel_size), index=[particle_id])], axis=1) if nb_labels > 0 else pd.DataFrame({'img_file_name': 'thumbnail_{}_{}.png'.format(str(sample_id).rstrip(), str(particle_id).rstrip()), 'Sample': sample_id, 'nb_particles': nb_labels}, index=[particle_id])
                            df_properties_sample_merged = pd.concat([df_properties_sample_merged, df_properties_git], axis=0).reset_index(drop=True)

                            # Split the mosaic according to the slices of rectangular regions to generate thumbnail and save
                            contour = ski.morphology.dilation(largest_object.astype(int), footprint=ski.morphology.square(3), out=None, shift_x=False, shift_y=False)
                            contour -= largest_object.astype(int)

                            scale_value = 50  # size of the scale bar in microns
                            padding = int((np.ceil(scale_value / pixel_size) + 10) / 2)
                            fig, axes = plt.subplots(1, 1)
                            ##fig.tight_layout(pad=0, h_pad=0, w_pad=-1)
                            # plt.imshow(np.lib.pad(contour, ((25, 25), (padding, padding)), constant_values=0), cmap='gray_r')
                            plt.imshow(np.lib.pad(colored_image[df_properties.at[rect_idx[id_of_interest] - 1, 'slice']][slice(1, -1, None), slice(1, -1, None)], ((25, 25), (padding, padding), (0, 0)),'constant', constant_values=float(df_metadata.at[sample, 'Background_Intensity_Mean'])),cmap='gray')

                            axes.set_axis_off()
                            scalebar = AnchoredSizeBar(transform=axes.transData, size=scale_value / pixel_size,
                                                       label='{} $\mu$m'.format(scale_value), loc='lower center',
                                                       pad=0.1,
                                                       color='white',
                                                       frameon=False,
                                                       fontproperties=fontprops)
                            axes.add_artist(scalebar)
                            # axes.set_title('Particle ID: {}'.format(str(particle_id)))
                            save_directory = Path(str(file.parent).replace('A repasser', 'outputs')).expanduser().parent / sample
                            save_directory.mkdir(parents=True, exist_ok=True)
                            fig.savefig(fname=str(save_directory / 'thumbnail_{}_{}.jpg'.format(str(sample).rstrip(),str(particle_id).rstrip())), transparent=False, bbox_inches="tight", pad_inches=0, dpi=300)
                            plt.close('all')

            bar.update(n=1)
    # Merge the de-novo properties datatable with FlowCam data, re-format for EcoTaxa and save
    df_properties_merged = pd.merge(df_properties_sample_merged, df_properties_sample.drop(columns=['Name']).rename(columns=dict(zip(df_properties_sample.columns, 'Visualspreadsheet ' + df_properties_sample.columns))).reset_index().astype({'Capture ID': str}).assign(Sample=sample), how='left', on=['Sample', 'Capture ID'])
    df_properties_all = pd.concat([df_properties_all, df_properties_merged], axis=0).reset_index(drop=True)
    filename_ecotaxa = str( save_directory.parent / save_directory.stem.rstrip().replace(' ', '_') / 'ecotaxa_table_{}.tsv'.format(str(sample).rstrip()))
    df_ecotaxa = generate_ecotaxa_table(df=pd.merge(df_properties_merged, df_volume.loc[sample].to_frame().T.assign(acq_acquisition_date=lambda x: pd.to_datetime(x.Start_Time, format='%Y-%m-%d %H:%M:%S').dt.floor('1d').dt.strftime('%Y%m%d'),Sample_Volume_Processed=lambda x: x.Sample_Volume_Processed.str.replace(' ml', '').astype(float), Sample_Volume_Aspirated=lambda x: x.Sample_Volume_Aspirated.str.replace(' ml', '').astype(float),Fluid_Volume_Imaged=lambda x: x.Fluid_Volume_Imaged.str.replace(' ml', '').astype(float), Flow_Rate=lambda x: x.Flow_Rate.str.replace(' ml/min', '').astype(float)).rename( columns={'Sample_Volume_Processed': 'sample_volume_analyzed_ml', 'Sample_Volume_Aspirated': 'sample_volume_pumped_ml','Fluid_Volume_Imaged': 'sample_volume_fluid_imaged_ml', 'Sampling_Time': 'sample_duration_sec', 'Flow_Rate': 'sample_flow_rate'})[['acq_acquisition_date', 'sample_volume_analyzed_ml', 'sample_volume_pumped_ml','sample_volume_fluid_imaged_ml', 'sample_duration_sec', 'sample_flow_rate']], how='left', right_index=True,left_on=['Sample']).assign(Sample=lambda x: x.Sample+'_xx_xx_'+x.Sample.str.split('_').str[0]), instrument='FlowCam', path_to_storage=filename_ecotaxa)


    #upload_thumbnails_ecotaxa_project(ecotaxa_configuration=configuration, project_id=int( cfg_metadata['ecotaxa_' + df_metadata.loc[sample, 'Ecotaxa_project'].lower() + '_projectid']),source_path=str(save_directory) + '.zip')

    # Generate Normalized Biovolume Size Spectra
    df_nbss_sample, df_nbss_boot_sample = nbss_estimates(df=pd.merge(df_properties_merged, df_volume.loc[sample].to_frame().T.assign( volume=lambda x: x.Fluid_Volume_Imaged.str.replace(' ml', '').astype(float))[['volume']], how='left', right_index=True, left_on=['Sample']), pixel_size=1, grouping_factor=[ 'Sample'])  # Set pixel size to 1 since pixel units were already converted to metric units
    df_nbss = pd.concat([df_nbss, df_nbss_sample], axis=0).reset_index(drop=True)
    # Append summary data to excel spreadsheet
    df_summary=df_properties_merged.assign(size_class=lambda x: pd.cut(x['Visualspreadsheet Diameter (ABD)'],np.array([0,10,20,50,150]),ordered=True)).size_class.value_counts().to_frame().T
    df_summary['Total'] = df_summary.sum(axis=1)
    df_summary[['Sample','Volume_processed_ml','Volume_imaged_ml']]=sample,df_ecotaxa.loc[1,'sample_volume_analyzed_ml'],df_ecotaxa.loc[1,'sample_volume_fluid_imaged_ml']
    df_summary.columns=df_summary.columns.astype(str).map({'Total':'Total','Sample':'Sample','Volume_processed_ml':'Volume_processed_ml','Volume_imaged_ml':'Volume_imaged_ml','(0, 10]':'0 to 10 µm','(10, 20]':'10 to 20 µm','(20, 50]':'20 to 50 µm','(50, 150]':'50 to 150 µm'})
    df_excel=pd.concat([df_excel,df_summary[['Sample','Volume_processed_ml','Volume_imaged_ml','0 to 10 µm','10 to 20 µm','20 to 50 µm','50 to 150 µm','Total']]],axis=0).reset_index(drop=True)
    with pd.ExcelWriter(str(path_to_network.parent / 'outputs' / 'Flowcam_recap_filtre_taille.xlsx'), engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_excel.to_excel(writer, sheet_name='New', index=False)

# Plot the Normalized Biovolume Size Spectra
#Attention, grouping factor should be a string
plot = (ggplot(df_nbss) +
        #geom_point(mapping=aes(x='(1/6)*np.pi*(size_class_mid**3)', y='NBSS'), alpha=1) +  #
        #stat_summary(data=df_nbss_boot_sample.melt(id_vars=['Group_index','Sample','size_class_mid'],value_vars='NBSS'),mapping=aes(x='size_class_mid', y='value',group='Sample',fill='Sample'),geom='ribbon',alpha=0.1,fun_data="median_hilow",fun_args={'confidence_interval':0.95})+
        #geom_ribbon(mapping=aes(x='size_class_mid', y='NBSS',ymin='np.maximum(0,NBSS-NBSS_std/2)',ymax='NBSS+NBSS_std/2',group='Sample',color='Sample'),alpha=0.1)+
        geom_point(mapping=aes(x='size_class_mid', y='NBSS',group='Group_index',colour='Sample'), alpha=1)+
        labs(x='Equivalent circular diameter ($\mu$m)',y='Normalized Biovolume Size Spectra ($\mu$m$^{3}$ mL$^{-1}$ $\mu$m$^{-3}$)', title='',colour='') +
        scale_y_log10(breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+04)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        scale_x_log10( breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e+00)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e+00)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(), labels=lambda l: [v if ((v / (10 ** np.floor(np.log10(v)))) == 1) else '' for v in l]) +
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)
plot.show()