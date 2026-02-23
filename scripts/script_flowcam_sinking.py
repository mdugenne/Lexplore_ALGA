# conda activate /Users/dugennem/opt/anaconda3/envs/xesmf_env
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as animation
from functools import partial
import pandas as pd
from pathlib import Path
import os
from natsort import natsorted
import skimage as ski  # pip install -U scikit-image
from skimage import color, measure, morphology
from PIL import Image, ImageColor
from skimage.util import compare_images, crop
from skimage.color import rgb2gray
from skimage.measure import regionprops
import cv2
import torch
from tqdm import tqdm
import numpy as np
import scipy as sp
import warnings
warnings.filterwarnings(action='ignore', category=SyntaxWarning)
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)
from plotnine import *

#Workflow starts here

path_to_dir=Path('~/GIT/Lexplore_ALGA/data/datafiles/flowcam/sinking').expanduser()
os.chdir(path_to_dir)
sys.path.append(str(path_to_dir))
from funcs_image_utils import *
import shutil

'''
from  tkinter import *
from  tkinter import filedialog
root = Tk()
root.directory = filedialog.askdirectory()
os.chdir(root.directory)
'''

path_to_runs=natsorted(list(path_to_dir.glob('*2026*')))
run=path_to_runs[1]
print('Processing run: {}'.format(run.stem))
df_properties_sample=pd.read_csv(run / str(run.name+'.csv'),sep=',',decimal='.', encoding='latin-1', index_col='Capture ID')
df_properties_sample['Time']=pd.to_datetime(df_properties_sample.loc[1,'Time'],format='%m-%d-%Y %H:%M:%S')+pd.to_timedelta(df_properties_sample['Elapsed Time'].astype(str).str.replace(',','.').astype(float), unit='s')
df_properties_sample['img_file_name'] = 'thumbnail_' + run.name + '_' + df_properties_sample.index.astype(str) + '.jpg'

df_context = pd.read_csv(list(path_to_dir.glob('*.ctx'))[-1], sep=r'\=|\t', engine='python', encoding='latin-1', names=['Name', 'Value']).query('not Name.str.contains(r"\[",case=True)').set_index('Name').T

# Load context file for cropping area
df_cropping = df_context[['AcceptableTop', 'AcceptableBottom', 'AcceptableLeft', 'AcceptableRight']]

## Downsizing the run by decreasing the temporal resolution (lowest image acauisition rate is & frame per second, which is too high for overnight runs)
## checkpoint and actual filtering will be done in cell #4
dowsizing=True # reset as False if downsizing the dataset is not necessary
new_fps=10 # interval in seconds between new frames
# Downsizing the dataset by decreasing in silico the image acquisition rate
if dowsizing:
    df_properties_sample['Elapsed_Time_reset']=df_properties_sample['Elapsed Time'].astype(str).str.replace(',','.').astype(float)
    df_properties_sample['Elapsed_Time_reset']=df_properties_sample['Elapsed_Time_reset']-df_properties_sample.loc[1,'Elapsed_Time_reset']
    selected_resolution=df_properties_sample['Elapsed_Time_reset'].round()==(df_properties_sample['Elapsed_Time_reset'].round()//new_fps)*new_fps
    df_properties_sample['selected_time_interval']=selected_resolution
    #df_properties_sample=df_properties_sample[selected_resolution].reset_index().set_index('Capture ID')
    id_to_skip = df_properties_sample.query('selected_time_interval==False').index.tolist()
    print("Downsizing dataset by decreasing the image acquisition rate to 1 frame per {} seconds.\nNew row number ({})".format(str(new_fps),str(len(df_properties_sample[selected_resolution]))))
else:
    selected_resolution=np.arange(len(df_properties_sample))
    id_to_skip = []
df_properties_sample

plot=(ggplot(df_properties_sample)+geom_point(mapping=aes(x='Time',y='Capture Y',colour='selected_time_interval',size='selected_time_interval'))+ scale_color_manual(values={False:'gray',True:'black'})+scale_size_manual(values={False:.001,True:1})+scale_y_reverse()).draw(show=True)


## Processing vignettes mosaic

# Search for mosaic files (all starting with ***collage***) in local data storage repository
mosaicfiles = natsorted(list(run.glob('collage_*.png')))

particle_id = 0
save_raw_directory = Path(run) / 'images'
save_ecotaxa_directory = Path(run) / 'ecotaxa'

background = ski.io.imread(run/'cal_image_000001.tif' ,as_gray=True)
import matplotlib.pyplot as plt
id_discarded = id_to_skip
# Check for existing vignettes, otherwise
if len(list(save_raw_directory.glob('*.jpg'))):
    id_to_skip=pd.Series(natsorted(list(set(df_properties_sample.img_file_name.unique()) - set(pd.Series(list(save_raw_directory.glob('*.jpg'))).astype(str).str.split(os.sep).str[-1].values)))).str.split('_').str[-1].str.replace('.jpg','').astype(int).unique()
    df_properties_sample = df_properties_sample.drop(index=id_to_skip)
else:
# Generate framed vignettes
    with tqdm(desc='Generating vignettes for run {}'.format(run.name), total=len(natsorted(mosaicfiles)), bar_format='{desc}{bar}', position=0, leave=True) as bar:
            for file in natsorted(mosaicfiles):

                file=Path(file).expanduser()
                percent = np.round(100 * (bar.n / len(mosaicfiles)), 1)
                bar.set_description('Generating vignettes for run {} (%s%%)'.format(run.name) % percent, refresh=True)

                pixel_size = float(0.7339)  # in microns per pixel


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
                rect_idx=df_properties.query('(area_convex==area_bbox) & (extent.round(3)==1)').index+1#df_properties.query('(area_convex==area_bbox) & (extent==1)').index[:-2]+1
                ##plt.figure(),plt.imshow(np.where(np.in1d(labelled,rect_idx).reshape(labelled.shape),image,0), cmap='gray'),plt.show()
                labelid_idx=((df_properties.query('(area_convex!=area_bbox) & (extent!=1)')).sort_values(['bbox-1','bbox-0']).index)+1#((df_properties.query('(area_convex!=area_bbox) & (extent!=1)')[:-4]).sort_values(['bbox-1','bbox-0']).index)+1 #label are sorted according to the x position
                ##plt.figure(),plt.imshow(np.where(np.in1d(labelled,labelid_idx).reshape(labelled.shape),image,0), cmap='gray'),plt.show()

                for id_of_interest in np.arange(0,len(rect_idx)):
                    if len(np.where(np.in1d(labelled,rect_idx[id_of_interest]).reshape(labelled.shape),image,0)[df_properties.at[rect_idx[id_of_interest]-1,'slice']][slice(1, -1, None), slice(1, -1, None)]):
                        particle_id = particle_id + 1

                        vignette_id=np.pad(np.where(np.in1d(labelled,rect_idx[id_of_interest]).reshape(labelled.shape),image,0)[df_properties.at[rect_idx[id_of_interest]-1,'slice']][slice(1, -1, None), slice(1, -1, None)],20,constant_values=0)
                        ##plt.figure(),plt.imshow(vignette_id, cmap='gray'),plt.show()
                        ##plt.figure(),plt.imshow(cv2.cvtColor(colored_image[df_properties.at[rect_idx[id_of_interest]-1,'slice']][slice(1, -1, None), slice(1, -1, None)].T, cv2.COLOR_BGR2RGB)),plt.show()


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
                            cropped_bottom=cropped_bottom+np.where(cropped_x==None,0,cropped_x)
                            cropped_right = cropped_right + np.where(cropped_y==None,0,cropped_y)
                            df_properties_sample.loc[particle_id,'Image Height']=df_properties_sample.loc[particle_id,'Image Height']+np.where(cropped_x==None,0,cropped_x)
                            df_properties_sample.loc[particle_id, 'Image Width'] =df_properties_sample.loc[particle_id, 'Image Width']+ np.where(cropped_y==None,0,cropped_y)


                        image_framed=background.copy()
                        image_framed[int(cropped_top):int(cropped_bottom),int(cropped_left):int(cropped_right)] = 255*image_cropped
                        image_framed[int(cropped_top):int(cropped_bottom),int(cropped_left):int(cropped_left)+1]=0
                        image_framed[int(cropped_top):int(cropped_bottom), int(cropped_right):int(cropped_right) + 1] = 0
                        image_framed[int(cropped_top):int(cropped_top)+1, int(cropped_left):int(cropped_right)] = 0
                        image_framed[int(cropped_bottom):int(cropped_bottom)+1, int(cropped_left):int(cropped_right)] = 0
                        #plt.figure(), plt.imshow(cv2.cvtColor(image_framed, cv2.COLOR_BGR2RGB)), plt.show()

                        # Discard ghost particles
                        diff_image = compare_images(image_cropped,background_cropped, method='diff')
                        ##plt.figure(),plt.imshow(diff_image),plt.show()
                        edges = ski.filters.sobel(diff_image)
                        markers = np.zeros_like(diff_image)
                        threshold=df_context[['ThresholdDark', 'ThresholdLight']]
                        markers[diff_image >(threshold.astype({'ThresholdDark': float, 'ThresholdLight': float})[ ['ThresholdDark', 'ThresholdLight']].min(axis=1).values / 255)] = 1
                        #markers[diff_image > (1/threshold.astype({'ThresholdDark':float, 'ThresholdLight':float})[['ThresholdDark', 'ThresholdLight']].max(axis=1).values)] = 1
                        #markers[edges > np.quantile(edges, 0.85)] = 1
                        markers[(sp.ndimage.binary_closing(edges) == False)] = 0
                        ##plt.figure(), plt.imshow(markers, cmap='gray'), plt.show()
                        fill_image = ski.morphology.erosion(ski.morphology.erosion(ski.morphology.closing(ski.morphology.dilation(markers,mode='constant')))) #
                        distance_neighbor=df_context[['DistanceToNeighbor']]
                        fill_image = ski.morphology.remove_small_holes(ski.morphology.closing(fill_image, ski.morphology.square(np.max([1,int(np.floor(distance_neighbor.DistanceToNeighbor.astype(float).values[0]/pixel_size/4))]))).astype(bool)) # int(np.floor(int(df_context.DistanceToNeighbor.astype(float).values[0]) / pixel_size))
                        ##plt.figure(), plt.imshow(fill_image, cmap='gray'), plt.show()
                        label_objects, nb_labels = sp.ndimage.label(fill_image)
                        min_esd=df_context[['MinESD']]
                        label_objects = ski.morphology.remove_small_objects(label_objects, min_size=float( min_esd.MinESD.values[0]) / pixel_size)
                        label_objects, nb_labels = sp.ndimage.label(label_objects) # Needs to re-assign the label after discarding small objects
                        if nb_labels<1:
                            id_to_skip=id_to_skip+[particle_id]

                        # Check if particle ID should be skipped
                        if particle_id not in id_to_skip:


                                #Image Saving raw thumbnail for CNN feature extraction
                                thumbnail=Image.fromarray(image_framed)#Image.fromarray(colored_image[df_properties.at[rect_idx[id_of_interest] - 1, 'slice']][slice(1, -1, None), slice(1, -1, None)])
                                save_raw_directory.mkdir(parents=True, exist_ok=True)
                                thumbnail.save(str(save_raw_directory / 'thumbnail_{}_{}.jpg'.format(str(file.parent.name).rstrip(), str(particle_id).rstrip())))
                                thumbnail_cropped = Image.fromarray(colored_image[df_properties.at[rect_idx[id_of_interest] - 1, 'slice']][slice(1, -1, None), slice(1, -1, None)])
                                save_ecotaxa_directory.mkdir(parents=True, exist_ok=True)
                                thumbnail_cropped.save(str(save_ecotaxa_directory / 'thumbnail_{}_{}.jpg'.format(str(file.parent.name).rstrip(),str(particle_id).rstrip())))

                        else:
                            id_to_skip=id_to_skip+[particle_id]

                bar.update(n=1)
    df_properties_sample=df_properties_sample.drop(index=np.unique(id_to_skip))
# Format and save properties table for Ecotaxa
filename_ecotaxa = str( save_ecotaxa_directory / 'ecotaxa_table_{}.tsv'.format( str(run.name).rstrip()))
df_ecotaxa = generate_ecotaxa_table(df=df_properties_sample, instrument='FlowCam', path_to_storage=filename_ecotaxa)

# Compress folder to prepare upload on Ecotaxa
shutil.make_archive(str(save_ecotaxa_directory), 'zip', save_ecotaxa_directory, base_dir=None)

print('New Ecotaxa zip file can be uploaded to the project\n{}'.format(str(save_ecotaxa_directory)+'.zip'))

# Extract features from CNN
save_features_directory=save_raw_directory.parent
if not Path(str(save_features_directory /'images_features.csv')).exists():
    df_features = image_feature_dataset(image_path=save_raw_directory,filter=df_properties_sample.img_file_name, model_name='efficientnet_b0', layer_name='avgpool')
    df_features
    df_features=df_features.copy()[df_features.ID.astype(str).isin(df_properties_sample.img_file_name.str.replace('.jpg','').unique())].reset_index(drop=True)
    df_features.to_csv(str(save_features_directory /'images_features.csv'),index=False)

else:
    df_features = pd.read_csv(str(save_features_directory / 'images_features.csv' ) )
# Project and Cluster image features
if not Path(str(save_features_directory /'images_features_projected.csv')).exists():
    model_cluster = TSNE(n_components=3)
    df_projection = pd.DataFrame(model_cluster.fit_transform(pd.DataFrame(normalize(df_features[df_features.columns[np.arange(2, df_features.shape[1])]]))))
    df_projection['image_url'] = df_features.path
    df_projection.rename(columns={'image_url': 'images'}).to_csv(str(save_features_directory / 'images_features_projected.csv' ) , index=False)
else:
    df_projection=pd.read_csv(str(save_features_directory / 'images_features_projected.csv' ))
# Check projected images with clusters
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)
import plotly.io as pio
pio.renderers.default = "browser"
app,df_clusters=scatter_2d_images(df_directory=str(save_features_directory / 'images_features_projected.csv' ))
app.run(port=8888, use_reloader=False,debug=True)
print('App running on local host: http://127.0.0.1:8888/.\nPress Control+c to quit')

# Plot projection
import plotnine
plotnine.options.base_margin=0.05
plot=(ggplot(df_clusters.melt(id_vars=['images','score','cluster','x'],value_vars=['y','z'],var_name='axis',value_name='value'))+facet_wrap('~axis')+labs(y='',x='',colour='Particle cluster:')+geom_point(mapping=aes(x='x',y='value',colour='factor(cluster)'),alpha=.2,size=5)+scale_color_manual(values=dict(zip(df_clusters.cluster.unique(),px.colors.qualitative.Plotly*int(np.ceil(df_clusters.cluster.nunique()/ 10)))))+theme_plot+
      theme(legend_position=(.5, -1.05),legend_direction='horizontal',plot_margin=1.7)+guides(color=guide_legend(direction='horizontal',nrow=3))).draw(show=True).set_size_inches(12,5)
plot.savefig(fname='{}/run_{}_projection.pdf'.format(str(save_features_directory),run.stem), dpi=300, bbox_inches='tight')


# Estimate sinking speed for each cluster(=particle) from linear regression between capture y-position and time
delta_seconds=1
if 'cluster' not in df_properties_sample.columns:
    df_properties_sample =pd.merge(df_properties_sample,df_clusters.assign(img_file_name=lambda x: x.images.astype(str).str.split(os.sep).str[-1])[['img_file_name','cluster']],how='left',on='img_file_name')
#Correction of sinking velocity for wall effects (according to the equation given by Ristow, 1997)
sinking_chamber_diameter=float(df_context.at['Value','FlowCellWidth']) # in micrometers
sinking_chamber_factor=1.004 #use 1.004 for two opposing walls ,use 2.1044 for a cylinder. Both values are only valid, if the particle is exactly in the middle of the sinking chamber
# Filter out cluster that contain only 6 or less repeated particles
df_properties_sample=df_properties_sample.loc[df_properties_sample.cluster.astype(str).isin( df_properties_sample.cluster.astype(str).value_counts().to_frame().index[np.where(df_properties_sample.cluster.value_counts() > 6)])]
# Returns dataframe including capture ID of tracked particles whose consecutive positions are recorded within delta_seconds
# linear regression (Y-position distance versus elapsed time in seconds) coefficients accounting for the camera resolution
# and trajectory coordinates (x,y) following the linear fit, corrected for the initial recording time
df_summary=pd.concat(list(map(lambda K:( df_subset:=df_properties_sample.query('cluster=={}'.format(str(K))).sort_values(['Time']).reset_index(drop=True),ind_to_keep:=np.arange(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0][np.argmax(np.diff(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)))],np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0][np.argmax(np.diff(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)))+1]) if len((np.where(df_subset.Time.diff().dt.seconds>delta_seconds))[0])>1 else list(set(np.arange(0,max(df_subset.index)+1))-set(pd.Series(''.join([indices for i,indices in pd.Series('-'.join(pd.Series(np.arange(0,max(df_subset.index)+1)).astype(str)).partition(str(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0][0]))).items() if i!=np.argmax(list(map(len,('-'.join(pd.Series(np.arange(0,max(df_subset.index)+1)).astype(str)).partition(str(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0][0]))))))]).split('-')).astype(int))) if len((np.where(df_subset.Time.diff().dt.seconds>delta_seconds))[0])==1 else list(set(np.arange(0,max(df_subset.index)+1))-set(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0].tolist())),ind_to_drop:=list(set(df_subset.index)-set(ind_to_keep)),df_subset:=df_subset.drop(index=ind_to_drop).reset_index(drop=True) if any(df_subset.Time.diff().dt.seconds>delta_seconds) else df_subset,regression:=stats.linregress(x=df_subset['Elapsed Time']-df_subset.loc[0,'Elapsed Time'], y=df_subset['Capture Y']*df_subset['Calibration Factor']),particles_id:=df_subset.img_file_name.str.split('_').str[-1].str.replace('.jpg',''),df_reg:=pd.concat([pd.DataFrame({'cluster':K,'capture_id':'-'.join(particles_id)},index=[0]),pd.DataFrame({'slope':regression.slope,'slope_corrected':regression.slope/(1-sinking_chamber_factor*(np.mean(df_subset['Diameter (ABD)']*df_subset['Calibration Factor']))/sinking_chamber_diameter),'intercept':regression.intercept,'r_value':regression.rvalue,'p_value':regression.pvalue,'slope_std':regression.stderr,'intercept_std':regression.intercept_stderr},index=[0]) ],axis=1),pd.merge(df_reg,pd.DataFrame({'x':df_subset.loc[0,'Time']+pd.to_timedelta((df_subset['Elapsed Time']-df_subset.loc[0,'Elapsed Time']).astype(float), unit='s'),'y':(df_reg.at[0,'intercept']+df_reg.at[0,'slope']*(df_subset['Elapsed Time']-df_subset.loc[0,'Elapsed Time']))/df_subset['Calibration Factor']}).assign(cluster=K),how='left',on='cluster'))[-1],df_properties_sample.dropna(subset='cluster').cluster.unique()))).reset_index(drop=True)
df_summary.to_csv(str(save_features_directory / 'images_trajectory_summary.csv') ,index=False)
df_summary

path_to_readme=Path(path_to_runs[0].parent / 'README_images_trajectory_summary.txt')
if not path_to_readme.exists():
    with open(str(path_to_readme), 'w') as file:
        file.write( "README file for Flowcam sinking trajectories summary datafiles (First created on February, 2026):\n\nThe run directories should contain a summary table of particles sinking trajectory recorded by the FlowCam\nEach *images_trajectory_summary* table includes individual tracked particle info, such as:\n\n-Cluster ID: Unique integer corresponding of the identifier of the cluster included similar particles, returned by the DBscan algorithm (Density-based spatial clustering, see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) applied to projected image deep features. 3D projections are performed using the t-SNE algorithm (t-distributed stochastic neighbor embedding, see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)\n\n-Capture ID (native format, separated by -): Unique capture ID, allowing to link the cluster to particles on the thumbnails and in the csv table including all hand-crafted features\n\n-slope (micrometers per second): Slope of the Capture-Y position (in micrometers) over elapsed time (in seconds). Should be multiplied by (1x10^-06)*(24*3600) to estimate the sinking speed in meters per day\n\n-slope_corrected (micrometers per second): Slope of the Capture-Y position (in micrometers) over elapsed time (in seconds), corrected for wall effects according to Bach et al. (2012, https://link.springer.com/article/10.1007/s00227-012-1945-2). Should be multiplied by (1x10^-06)*(24*3600) to estimate the corrected sinking speed in meters per day\n\n-intercept (micrometers): Y-capture position at initial particle location\n\n-r_value (unitless): Goodness of the linear fit determining the slope\n\n-p_value (unitless): Probability that the slope is different zero by chance only\n\n-slope_std (micrometer per second):  Standard deviation of the slope of the Capture-Y position (in micrometers) over elapsed time (in seconds) linear fit\n\n-intercept_std (micrometer):  Standard deviation of the Capture-Y position (in micrometers) initial position computed from the linear fit\n\n-x (in seconds): Elapsed time since acquisition start\n\n-y (in micrometers): Predicted Y-Capture position in the flow cell])\n\n\nFor questions, contact: mathilde.dugenne@unige.ch")

plot=(ggplot(df_properties_sample)+labs(y='Y-axis position in the frame',x='Recording time',colour='Particle cluster:')+geom_point(mapping=aes(x='Time',y='Capture Y',colour='factor(cluster)'),alpha=.2,size=5)+geom_line(data=df_summary,mapping=aes(x='x',y='y',group='factor(cluster)'))+scale_y_reverse()+scale_color_manual(values=dict(zip(df_clusters.cluster.unique(),px.colors.qualitative.Plotly*int(np.ceil(df_clusters.cluster.nunique()/ 10)))))+theme_plot+theme(legend_title_position='top')+theme(legend_position=(.5, -1.05),legend_direction='horizontal',plot_margin=1.7)+guides(color=guide_legend(direction='horizontal',nrow=3))).draw(show=True)

(df_summary[['cluster','slope','slope_corrected']].drop_duplicates().set_index('cluster').slope*1e-06)*(24*3600)
(df_summary[['cluster','slope','slope_corrected']].drop_duplicates().set_index('cluster').slope_corrected*1e-06)*(24*3600)
df_sinking_speed=((df_summary[['cluster','slope','slope_corrected']].drop_duplicates().set_index('cluster')[['slope','slope_corrected']]*1e-06)*(24*3600)).reset_index().melt(id_vars='cluster', value_vars=['slope','slope_corrected'], var_name='coefficient', value_name='value')
plot=(ggplot(df_sinking_speed[(df_sinking_speed.value>=np.quantile(df_sinking_speed.value.dropna(),0.05)) & (df_sinking_speed.value<=np.quantile(df_sinking_speed.value.dropna(),0.95))])+labs(y='Number of observations',x='Sinking speed (m/d)',fill='')+geom_histogram(mapping=aes(x='value',fill='coefficient'),alpha=.5,colour='black',bins=50)+scale_fill_manual(values={'slope':'gray','slope_corrected':'blue'})+theme_plot+theme(legend_position='top',legend_direction='horizontal',legend_title_position='top')).draw(show=True)


df_hist=(df_sinking_speed.astype({'cluster':str}).assign(value_binned=lambda x: pd.cut(x.value,bins=50)).assign(value_binned=lambda x: x.value_binned.map(dict(zip(x.value_binned.cat.categories,pd.IntervalIndex(x.value_binned.cat.categories).mid))))).astype({'value_binned':str}).groupby(['coefficient','value_binned'],observed=True).apply(lambda x: x.value_binned.value_counts().to_frame().reset_index().assign(cluster='-'.join(x.cluster.unique())).drop(columns=['value_binned'])).reset_index().astype({'value_binned':float}).sort_values(['coefficient','value_binned']).dropna().reset_index(drop=True).drop(columns=['level_2'])
fig=px.bar(df_hist,x='value_binned',y='count',color='coefficient',color_discrete_sequence =['gray','blue'],hover_data={'cluster':True,'value_binned':True,'coefficient':True})
fig.update_layout(legend_title=dict(text=''),title=dict(text='Sinking speed distribution'),yaxis_title=dict(text='Number of observations'),xaxis_title=dict(text='Sinking speed (m/d)'),xaxis=dict(showgrid=False, showline=True, linecolor='black'),yaxis=dict(showgrid=False, showline=True, linecolor='black'),plot_bgcolor='rgba(255,255,255,0)')
fig.show()


# Generate animated frames
df_summary=df_summary.astype({'cluster':str}).dropna(subset='slope')
fig, ax = plt.subplots(nrows=2,height_ratios=[3,1])
with tqdm(desc='Generating reconstructed frames for cluster', total=df_summary.cluster.nunique(), bar_format='{desc}{bar}', position=0, leave=True) as bar:
    for K in df_summary.cluster.unique():
        bar.set_description('Generating reconstructed frames for cluster {}'.format(K) , refresh=True)
        if len(df_summary.query('cluster=="{}"'.format(K)).capture_id.unique()):
            n_frames=len(df_summary.query('cluster=="{}"'.format(K)).capture_id.unique()[0].split('-'))
            if n_frames>4:
                anim = animation.FuncAnimation(fig, partial(animate, cluster=K, df_properties=df_properties_sample.astype({'cluster': str}),df_regression=df_summary.astype({'cluster': str}),df_clusters=df_clusters.astype({'cluster': str})),frames=n_frames, blit=False, repeat=False)
                anim.save( save_raw_directory.parent/ 'animation_framed_cluster_{}.gif'.format(str(int(K))), fps=5)
        bar.update(n=1)

# For testing
K=0
df_subset=df_properties_sample.query('cluster=={}'.format(str(K))).sort_values(['Time']).reset_index(drop=True)
df_subset.Time
df_subset['diff_Time']=df_subset.Time.diff().dt.seconds
np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0]
np.diff(np.where(df_subset.Time.diff().dt.seconds>delta_seconds))
ind_to_keep=np.arange(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0][np.argmax(np.diff(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)))],np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0][np.argmax(np.diff(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)))+1]) if len((np.where(df_subset.Time.diff().dt.seconds>delta_seconds))[0])>1 else list(set(np.arange(0,max(df_subset.index)+1))-set(pd.Series(''.join([indices for i,indices in pd.Series('-'.join(pd.Series(np.arange(0,max(df_subset.index)+1)).astype(str)).partition(str(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0][0]))).items() if i!=np.argmax(list(map(len,('-'.join(pd.Series(np.arange(0,max(df_subset.index)+1)).astype(str)).partition(str(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0][0]))))))]).split('-')).astype(int))) if len((np.where(df_subset.Time.diff().dt.seconds>delta_seconds))[0])==1 else list(set(np.arange(0,max(df_subset.index)+1))-set(np.where(df_subset.Time.diff().dt.seconds>delta_seconds)[0].tolist()))
ind_to_drop=list(set(df_subset.index)-set(ind_to_keep))
df_subset=df_subset.drop(index=ind_to_drop).reset_index(drop=True) if any(df_subset.Time.diff().dt.seconds>5) else df_subset
regression=stats.linregress(x=df_subset['Elapsed Time']-df_subset.loc[0,'Elapsed Time'], y=df_subset['Capture Y']*df_subset['Calibration Factor'])
df_reg=pd.concat([pd.DataFrame({'cluster':K},index=[0]),pd.DataFrame({'slope':regression.slope,'intercept':regression.intercept,'r_value':regression.rvalue,'p_value':regression.pvalue,'slope_std':regression.stderr,'intercept_std':regression.intercept_stderr},index=[0]) ],axis=1)
df_summary=pd.DataFrame({'x':df_subset.loc[0,'Time']+pd.to_timedelta((df_subset['Elapsed Time']-df_subset.loc[0,'Elapsed Time']).astype(float), unit='s'),'y':(df_reg.at[0,'intercept']+df_reg.at[0,'slope']*(df_subset['Elapsed Time']-df_subset.loc[0,'Elapsed Time']))/df_subset['Calibration Factor']}).assign(cluster=K)
df_reg.slope/(1-sinking_chamber_factor*(np.mean(df_subset['Diameter (ABD)']*df_subset['Calibration Factor']))/sinking_chamber_diameter)