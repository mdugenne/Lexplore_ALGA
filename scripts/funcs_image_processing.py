## Objective: This script includes all modules and functions required for image processing

import warnings
warnings.filterwarnings(action='ignore')

# Image processing modules
import skimage as ski #pip install -U scikit-image
from skimage import color,measure,morphology
from PIL import Image
from skimage.util import compare_images,crop
from skimage.color import rgb2gray
from skimage.measure import regionprops
import cv2

# Path and File processing modules
import os
import scipy as sp
from pathlib import Path
path_to_git=Path('~/GIT/Lexplore_ALGA').expanduser()
# Read the metadata stored in yaml file
import yaml #conda install PyYAML
path_to_config = path_to_git / 'data' / 'Lexplore_metadata.yaml'
with open(path_to_config, 'r') as config_file:
    cfg_metadata = yaml.safe_load(config_file)
import shutil # zip folder
import re

# Data processing modules
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm
from functools import reduce

# Plot modules
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
fontprops = fm.FontProperties(size=14,family='serif')
import copy
my_cmap = copy.copy(plt.colormaps.get_cmap('gray_r')) # get a copy of the gray color map
my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values
import latex

from plotnine import *
import plotnine
import matplotlib
from plotnine import geom_errorbar, coord_fixed, scale_fill_distiller
from plotnine.themes.themeable import legend_position

theme_paper=theme(panel_grid=element_blank(), legend_position='bottom',
              panel_background=element_rect(fill='#ffffff'), legend_background=element_rect(fill='#ffffff'),
              strip_background=element_rect(fill='#ffffff'),
              panel_border=element_rect(color='#222222'),
              legend_title=element_text(family='Times New Roman', size=12),
              legend_text=element_text(family='Times New Roman', size=12),
              axis_title=element_text(family='Times New Roman', size=12, linespacing=1),
              axis_text_x=element_text(family='Times New Roman', size=12, linespacing=0),
              axis_text_y=element_text(family='Times New Roman', size=12, rotation=90, linespacing=1),
              plot_background=element_rect(fill='#ffffff00'))
import seaborn as sns
summary_metadata_columns=['Mode', 'Priming_Method', 'Flow_Rate', 'Recalibrations', 'Stop_Reason',
       'Sample_Volume_Aspirated', 'Sample_Volume_Processed',
       'Fluid_Volume_Imaged', 'Efficiency', 'Particle_Count', 'Total', 'Used',
       'Percentage_Used', 'Particles_Per_Image', 'Frame_Rate',
       'Background_Intensity_Mean', 'Background_Intensity_Min',
       'Background_Intensity_Max', 'Start_Time', 'Sampling_Time',
       'Environment', 'Software', 'Magnification', 'Calibration_Factor',
       'SerialNo', 'Number_of_Processors', 'Pump', 'Syringe_Size','Skip']


# Size spectrum processing
from scipy.stats import poisson
from scipy import stats
import statsmodels
from statsmodels import formula
from statsmodels.formula import api
bins=np.power(2, np.arange(0, np.log2(100000) + 1 / 3, 1 / 3))  # Fixed size(micrometers) bins used for UVP/EcoPart data. See https://ecopart.obs-vlfr.fr/. 1/3 ESD increments allow to bin particle of doubled biovolume in consecutive bins. np.exp(np.diff(np.log((1/6)*np.pi*(EcoPart_extended_bins**3))))

df_bins = pd.DataFrame({'size_class': pd.cut(bins, bins).categories.values,  # Define bin categories (um)
                        'bin_widths': np.diff(bins),  # Define width of individual bin categories (um)
                        'range_size_bin': np.concatenate(np.diff((1 / 6) * np.pi * (np.resize( np.append(bins[0], np.append(np.repeat(bins[1:-1], repeats=2), bins[len(bins) - 1])),(len(bins) - 1, 2)) ** 3), axis=1)),  # cubic micrometers
                        'size_class_mid': stats.gmean(np.resize( np.append(bins[0], np.append(np.repeat(bins[1:-1], repeats=2), bins[len(bins) - 1])), (len(bins) - 1, 2)), axis=1)})  # Define geometrical mean of bin categories (micrometers)
def resample_biovolume(x,pixel,resample=False):
    n=np.random.poisson(lam=len(x))
    return np.abs(sum(np.random.normal(size=n,loc=np.mean(np.random.choice(x,size=n,replace=True)), scale=(1 / 6) * np.pi * (2 * ((1/(pixel*1e-06)) / np.pi) ** 0.5) ** 3))) if resample else sum(x)


def nbss_estimates(df,pixel_size,grouping_factor=['Sample'],niter=100):

    """
    Objective: This function computes the Normalized Biovolume Size Spectrum (NBSS) of individual grouping_factor levels from area estimates.\n
    The function also returns boostrapped NBSS based on the method published in Schartau et al. (2010)
    :param df: A dataframe of morphometric properties including *area* estimates for individual particles
    :param pixel_size: The camera resolution as defined by the manufacturer or calibrated with beads
    :param grouping_factor: A list including all possible variables that should be included to compute level-specific NBSS (e.g. sampling depth, taxonomic group)
    :param n_iter: The number of resampling iterations used to assess NBSS uncertainties. Default is 100 iterations
    :return: Returns a dataframe with NBSS and NBSS standard deviations based on the boostrapped NBSS and a dataframe with all boostrapped NBSS
    """

    # Assign a group index before groupby computation
    group = grouping_factor
    df_subset = pd.merge(df, df.drop_duplicates(subset=group, ignore_index=True)[group].reset_index().rename( {'index': 'Group_index'}, axis='columns'), how='left', on=group)
    # Assign size bins according to biovolume and equivalent circular diameter estimates
    df_subset=df_subset.assign(Pixel=1e+03/pixel_size,Area=lambda x:(x.area*(pixel_size**2)),Biovolume=lambda x:(1/6)*np.pi*(2*((x.area*(pixel_size**2))/np.pi)**0.5)**3,ECD=lambda x:2*((x.area*(pixel_size**2))/np.pi)**0.5)
    df_subset=df_subset.assign(size_class=lambda x:pd.cut(x.ECD,bins,include_lowest=True).astype(str))
    df_subset=pd.merge(df_subset,df_bins.astype({'size_class':str}),how='left',on=['size_class'])
    df_subset=df_subset.assign(size_class_pixel=lambda z:z.size_class_mid/pixel_size,ROI_number=1)

    # Pivot table after summing all biovolume for each size class
    x_pivot=df_subset.pivot_table(values='Biovolume',columns=['size_class_mid','range_size_bin','size_class'],index=grouping_factor+['Group_index','volume'],aggfunc=lambda z:resample_biovolume(z,pixel=1e+03/pixel_size,resample=False))
    # Normalize to size class width
    x_pivot=x_pivot/x_pivot.columns.levels[1]
    # Normalize to volume imaged
    size_class=x_pivot.columns.levels[2]
    size_class =size_class.drop('nan') if 'nan' in size_class else size_class
    x_pivot=(x_pivot.fillna(0)/pd.DataFrame(np.resize(np.repeat(x_pivot.index.get_level_values('volume'),x_pivot.shape[1]),(x_pivot.shape[0],x_pivot.shape[1])),index=x_pivot.index,columns=x_pivot.columns)).reset_index(col_level=2).droplevel(level=[0,1],axis=1)
    # Append number of observations and pixel size
    x_nbr=df_subset.pivot_table(values='ROI_number',columns=['size_class'],index=grouping_factor+['Group_index','volume'],aggfunc='sum')
    x_nbr=(x_nbr.fillna(0).cumsum(axis=0)).reset_index()
    # Melt datatable and sort by grouping factors and size classes
    x_melt=pd.merge((x_pivot.melt(id_vars=grouping_factor+['Group_index','volume'],value_vars=size_class,var_name='size_class',value_name='NBSS').replace({'NBSS':0}, np.nan)).astype({'size_class':str}),df_bins.astype({'size_class':str}),how='left',on='size_class').sort_values(grouping_factor+['Group_index','volume']+['size_class_mid']).reset_index(drop=True)
    x_melt=pd.merge(x_melt,x_nbr.melt(id_vars=grouping_factor+['Group_index','volume'],value_vars=size_class,var_name='size_class',value_name='NBSS_count').replace({'NBSS_count':0}, np.nan).astype({'size_class':str}),how='left',on=grouping_factor+['Group_index','volume','size_class'])
    x_melt=pd.merge(x_melt,((2*(df_subset.astype({'size_class':str}).groupby(['size_class']).Area.mean()/np.pi)**0.5/(df_subset.Pixel.unique()[0]*1e-03)).reset_index().rename(columns={'Area':'size_class_pixel'})),how='left',on='size_class')
    # Append boostrapped summaries
    df_nbss_std=list(map(lambda iter:(x_pivot:=df_subset.pivot_table(values='Biovolume',columns=['size_class_mid','range_size_bin','size_class'],index=grouping_factor+['Group_index','volume'],aggfunc=lambda z:resample_biovolume(z,pixel=df_subset.Pixel.unique()[0],resample=True)),x_pivot:=x_pivot/x_pivot.columns.levels[1],   x_pivot:=(x_pivot.fillna(0)/pd.DataFrame(np.resize(np.repeat(x_pivot.index.get_level_values('volume'),x_pivot.shape[1]),(x_pivot.shape[0],x_pivot.shape[1])),index=x_pivot.index,columns=x_pivot.columns)).reset_index(col_level=2).droplevel(level=[0,1],axis=1).melt(id_vars=grouping_factor+['Group_index','volume'],value_vars=size_class,var_name='size_class',value_name='NBSS').replace({'NBSS':0}, np.nan).sort_values(grouping_factor+['Group_index','volume']).reset_index(drop=True))[-1],np.arange(niter)))
    df_nbss_boot=reduce(lambda left, right: pd.merge(left, right, on=grouping_factor+['Group_index','volume','size_class'],suffixes=(None,'_right')),df_nbss_std).reset_index(drop=True).sort_values(grouping_factor+['Group_index']).rename(columns={'NBSS_right':'NBSS'})
    df_summary=pd.concat([df_nbss_boot.reset_index().drop(columns='NBSS'),pd.DataFrame({'NBSS_std':df_nbss_boot[['NBSS']].std(axis=1)})],axis=1)
    df_nbss_boot = pd.merge(df_nbss_boot, df_bins.astype({'size_class': str}), how='left', on=['size_class'])
    x_melt = pd.merge(x_melt,df_summary , how='left', on=grouping_factor+['Group_index','volume','size_class'])

    return x_melt,df_nbss_boot

def as_float_cytosense_depth(str):
    """
    Objective: This function is used to retrieve the sampling depth from the name of CytoSense datafiles (no other export options). Default is 0 unless the Depth Profile Sampler was activated
    """

    try:
        return float(str)
    except:
        return 0

# Define a dictionary to specify the columns types returned by the region of interest properties table
dict_properties_types={'nb_particles':['[f]'], 'area':['[f]'], 'area_bbox':['[f]'],
       'area_convex':['[f]'], 'area_filled':['[f]'], 'axis_major_length':['[f]'], 'axis_minor_length':['[f]'],
       'bbox-0':['[f]'], 'bbox-1':['[f]'], 'bbox-2':['[f]'], 'bbox-3':['[f]'], 'centroid_local-0':['[f]'],
       'centroid_local-1':['[f]'], 'centroid_weighted_local-0':['[f]'],
       'centroid_weighted_local-1':['[f]'], 'eccentricity':['[f]'], 'equivalent_diameter_area':['[f]'],
       'extent':['[f]'], 'image_intensity':['[t]'], 'inertia_tensor-0-0':['[f]'], 'inertia_tensor-0-1':['[f]'],
       'inertia_tensor-1-0':['[f]'], 'inertia_tensor-1-1':['[f]'], 'inertia_tensor_eigvals-0':['[f]'],
       'inertia_tensor_eigvals-1':['[f]'], 'intensity_mean':['[f]'], 'intensity_max':['[f]'],
       'intensity_min':['[f]'], 'intensity_std':['[f]'], 'moments-0-0':['[f]'], 'moments-0-1':['[f]'],
       'moments-0-2':['[f]'], 'moments-0-3':['[f]'], 'moments-1-0':['[f]'], 'moments-1-1':['[f]'],
       'moments-1-2':['[f]'], 'moments-1-3':['[f]'], 'moments-2-0':['[f]'], 'moments-2-1':['[f]'],
       'moments-2-2':['[f]'], 'moments-2-3':['[f]'], 'moments-3-0':['[f]'], 'moments-3-1':['[f]'],
       'moments-3-2':['[f]'], 'moments-3-3':['[f]'], 'moments_central-0-0':['[f]'],
       'moments_central-0-1':['[f]'], 'moments_central-0-2':['[f]'], 'moments_central-0-3':['[f]'],
       'moments_central-1-0':['[f]'], 'moments_central-1-1':['[f]'], 'moments_central-1-2':['[f]'],
       'moments_central-1-3':['[f]'], 'moments_central-2-0':['[f]'], 'moments_central-2-1':['[f]'],
       'moments_central-2-2':['[f]'], 'moments_central-2-3':['[f]'], 'moments_central-3-0':['[f]'],
       'moments_central-3-1':['[f]'], 'moments_central-3-2':['[f]'], 'moments_central-3-3':['[f]'],
       'moments_hu-0':['[f]'], 'moments_hu-1':['[f]'], 'moments_hu-2':['[f]'], 'moments_hu-3':['[f]'],
       'moments_hu-4':['[f]'], 'moments_hu-5':['[f]'], 'moments_hu-6':['[f]'], 'num_pixels':['[f]'],
       'orientation':['[f]'], 'perimeter':['[f]'], 'slice':['[t]']}
dict_properties_types_colors=dict_properties_types | {'centroid_weighted_local-0-0':['[f]'],'centroid_weighted_local-0-1':['[f]'],'centroid_weighted_local-0-2':['[f]'], 'centroid_weighted_local-1':['[f]'],'centroid_weighted_local-1-1':['[f]'],'centroid_weighted_local-1-2':['[f]'], 'intensity_mean':['[f]'], 'intensity_max':['[f]'], 'intensity_min':['[f]'], 'intensity_std':['[f]'], 'intensity_mean-0':['[f]'], 'intensity_max-0':['[f]'], 'intensity_min-0':['[f]'], 'intensity_std-0':['[f]'],'intensity_mean-1':['[f]'], 'intensity_max-1':['[f]'], 'intensity_min-1':['[f]'], 'intensity_std-1':['[f]'],'intensity_mean-2':['[f]'], 'intensity_max-2':['[f]'], 'intensity_min-2':['[f]'], 'intensity_std-2':['[f]']}
dict_properties_visual_spreadsheet={'Name':['[t]'], 'Area (ABD)':['[f]'], 'Area (Filled)':['[f]'], 'Aspect Ratio':['[f]'], 'Average Blue':['[f]'],
       'Average Green':['[f]'], 'Average Red':['[f]'], 'Calibration Factor':['[f]'],
       'Calibration Image':['[f]'], 'Capture ID':['[t]'] ,'Capture X':['[f]'], 'Capture Y':['[f]'], 'Circularity':['[f]'],
       'Circularity (Hu)':['[f]'], 'Compactness':['[f]'], 'Convex Perimeter':['[f]'], 'Convexity':['[f]'],
       'Date':['[t]'], 'Diameter (ABD)':['[f]'], 'Diameter (ESD)':['[f]'], 'Diameter (FD)':['[f]'],
       'Edge Gradient':['[f]'], 'Elapsed Time':['[f]'], 'Elongation':['[f]'], 'Feret Angle Max':['[f]'],
       'Feret Angle Min':['[f]'], 'Fiber Curl':['[f]'], 'Fiber Straightness':['[f]'], 'Filter Score':['[f]'],
       'Geodesic Aspect Ratio':['[f]'], 'Geodesic Length':['[f]'], 'Geodesic Thickness':['[f]'],
       'Group ID':['[t]'], 'Image Height':['[f]'], 'Image Width':['[f]'], 'Image X':['[f]'], 'Image Y':['[f]'],
       'Length':['[f]'], 'Particles Per Chain':['[f]'], 'Perimeter':['[f]'], 'Ratio Blue/Green':['[f]'],
       'Ratio Red/Blue':['[f]'], 'Ratio Red/Green':['[f]'], 'Roughness':['[f]'], 'Source Image':['[f]'],
       'Sum Intensity':['[f]'], 'Symmetry':['[f]'], 'Time':['[t]'], 'Transparency':['[f]'], 'Volume (ABD)':['[f]'],
       'Width':['[f]']}
#Define a metadata table for Lexplore based on the metadata confiugration file
df_sample_lexplore_cytosense=pd.DataFrame({'sample_longitude':['[f]',cfg_metadata['longitude']],'sample_latitude':['[f]',cfg_metadata['latitude']],'sample_platform':['[t]',cfg_metadata['program']],'sample_project':['[t]',cfg_metadata['project']],'sample_principal_investigator':['[t]',cfg_metadata['principal_investigator']],'sample_depthprofile':['[t]','True'] })
df_acquisition_lexplore_cytosense=pd.DataFrame({'acq_instrument':['[t]','CytoSense_CS-2015-71'],'acq_pumptype':['[t]','peristaltic'],'acq_software':['[t]',cfg_metadata['version_cytoUSB']],'acq_pixel_um':['[f]',cfg_metadata['pixel_size_cytosense']],'acq_frequency_fps':['[f]',cfg_metadata['fps_cytosense']],'acq_max_width_um':['[f]',cfg_metadata['width_cytosense']*cfg_metadata['pixel_size_cytosense']],'acq_max_height_um':['[f]',cfg_metadata['height_cytosense']*cfg_metadata['pixel_size_cytosense']]})
df_processing_lexplore_cytosense=pd.DataFrame({'process_operator':['[t]',cfg_metadata['instrument_operator']],'process_code':['[t]','https://github.com/mdugenne/Lexplore_ALGA'],'process_min_diameter_um':['[f]',10],'process_max_diameter_um':['[f]',cfg_metadata['width_cytosense']*cfg_metadata['pixel_size_cytosense']],'process_gamma':['[f]',0.7]})

df_context_flowcam_micro=pd.read_csv(r'{}'.format(cfg_metadata['flowcam_10x_context_file']),sep=r'\=|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains(r"\[",case=True)').set_index('Name').T.rename(index={'Value':Path(cfg_metadata['flowcam_10x_context_file']).stem})
df_context_flowcam_macro=pd.read_csv(r'{}'.format(cfg_metadata['flowcam_macro_context_file']),sep=r'\=|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains(r"\[",case=True)').set_index('Name').T.rename(index={'Value':Path(cfg_metadata['flowcam_macro_context_file']).stem})


df_sample_lexplore_flowcam=pd.DataFrame({'sample_longitude':['[f]',cfg_metadata['longitude']],'sample_latitude':['[f]',cfg_metadata['latitude']],'sample_platform':['[t]',cfg_metadata['program']],'sample_project':['[t]',cfg_metadata['project']],'sample_principal_investigator':['[t]',cfg_metadata['principal_investigator']],'sample_depthprofile':['[t]','False'],'sample_storage_gear':['[t]','wasam'],'sample_fixative':['[t]','glutaraldehyde'],'sample_fixative_final_concentration':['[f]',0.25] })
df_acquisition_lexplore_flowcam_10x=pd.DataFrame({'acq_instrument':['[t]','Flowcam_SN_{}'.format(df_context_flowcam_micro.SerialNo.astype(str).values[0])],'acq_software':['[t]',df_context_flowcam_micro.SoftwareName.astype(str).values[0]+'_v'+df_context_flowcam_micro.SoftwareVersion.astype(str).values[0]],'acq_flowcell':['[t]',df_context_flowcam_micro.FlowCellType.astype(str).values[0]],'acq_objective':['[t]','{}x'.format(df_context_flowcam_micro.CameraMagnification.astype(str).values[0])],'acq_mode':['[t]','Autoimage'],'acq_stop_criterion':['[t]','Volume_pumped'],'acq_pumptype':['[t]','syringe'],'acq_max_width_um':['[f]',(df_context_flowcam_micro.AcceptableRight.astype(float).values[0]-df_context_flowcam_micro.AcceptableLeft.astype(float).values[0])*cfg_metadata['pixel_size_flowcam_10x']],'acq_max_height_um':['[f]',(df_context_flowcam_micro.AcceptableBottom.astype(float).values[0]-df_context_flowcam_micro.AcceptableTop.astype(float).values[0])*cfg_metadata['pixel_size_flowcam_10x']],'acq_min_diameter_um':['[f]',df_context_flowcam_micro.MinESD.astype(float).values[0]],'acq_max_diameter_um':['[f]',df_context_flowcam_micro.MaxESD.astype(float).values[0]],'acq_dark_threshold':['[f]',df_context_flowcam_micro.ThresholdDark.astype(float).values[0]],'acq_dark_threshold':['[f]',df_context_flowcam_micro.ThresholdDark.astype(float).values[0]],'acq_light_threshold':['[f]',df_context_flowcam_micro.ThresholdLight.astype(float).values[0]],'acq_neighbor_distance':['[f]',df_context_flowcam_micro.DistanceToNeighbor.astype(float).values[0]],'acq_closing_iterations':['[f]',df_context_flowcam_micro.CloseHoles.astype(float).values[0]],'acq_rolling_calibration':['[f]',df_context_flowcam_micro.FrameCount.astype(float).values[0]]})
df_processing_lexplore_flowcam_10x=pd.DataFrame({'process_operator':['[t]',cfg_metadata['instrument_operator']],'process_code':['[t]','https://github.com/mdugenne/Lexplore_ALGA'],'process_imagetype':['[t]','mosaic']})
df_acquisition_lexplore_flowcam_macro=pd.DataFrame({'acq_instrument':['[t]','Flowcam_SN_{}'.format(df_context_flowcam_macro.SerialNo.astype(str).values[0])],'acq_software':['[t]',df_context_flowcam_macro.SoftwareName.astype(str).values[0]+'_v'+df_context_flowcam_macro.SoftwareVersion.astype(str).values[0]],'acq_flowcell':['[t]',df_context_flowcam_macro.FlowCellType.astype(str).values[0]],'acq_objective':['[t]','{}x'.format(df_context_flowcam_macro.CameraMagnification.astype(str).values[0])],'acq_mode':['[t]','Autoimage'],'acq_stop_criterion':['[t]','Volume_pumped'],'acq_pumptype':['[t]','syringe'],'acq_max_width_um':['[f]',(df_context_flowcam_macro.AcceptableRight.astype(float).values[0]-df_context_flowcam_macro.AcceptableLeft.astype(float).values[0])*cfg_metadata['pixel_size_flowcam_macro']],'acq_max_height_um':['[f]',(df_context_flowcam_macro.AcceptableBottom.astype(float).values[0]-df_context_flowcam_macro.AcceptableTop.astype(float).values[0])*cfg_metadata['pixel_size_flowcam_macro']],'acq_min_diameter_um':['[f]',df_context_flowcam_macro.MinESD.astype(float).values[0]],'acq_max_diameter_um':['[f]',df_context_flowcam_macro.MaxESD.astype(float).values[0]],'acq_dark_threshold':['[f]',df_context_flowcam_macro.ThresholdDark.astype(float).values[0]],'acq_dark_threshold':['[f]',df_context_flowcam_macro.ThresholdDark.astype(float).values[0]],'acq_light_threshold':['[f]',df_context_flowcam_macro.ThresholdLight.astype(float).values[0]],'acq_neighbor_distance':['[f]',df_context_flowcam_macro.DistanceToNeighbor.astype(float).values[0]],'acq_closing_iterations':['[f]',df_context_flowcam_macro.CloseHoles.astype(float).values[0]],'acq_rolling_calibration':['[f]',df_context_flowcam_macro.FrameCount.astype(float).values[0]]})
df_processing_lexplore_flowcam_macro=pd.DataFrame({'process_operator':['[t]',cfg_metadata['instrument_operator']],'process_code':['[t]','https://github.com/mdugenne/Lexplore_ALGA'],'process_imagetype':['[t]','mosaic']})

def generate_ecotaxa_table(df,instrument,path_to_storage=None):
    """
    Objective: This function generates a table to be uploaded on Ecotaxa along with the thumbnails after acquisitions with the CytoSense or FlowCam imaging flow cytometer
    :param df: A dataframe of morphometric properties including *area* estimates for individual particles. \nFor FlowCam acquisitions, this tables corresponds to the "data" csv file that can be exported automatically (in context file, go to the Reports tab and enable option "Export list data when run terminates") or manually (in the main window, click on the File tab and select "Export data". Note that if the run was closed you'll need to re-open the data by clicking on "Open data" amd selecting the run).
    \nFor CytoSense acquisitions, the table is generated by running the custom script_cytosense.py. This script uses cropped images (without scalebars) exported automatically after each acquisition.
    :param instrument: Instrument used for the acquisition. Supported options are "CytoSense" or "FlowCam"
    :param path_to_storage: A Path object specifying where the table should be saved. If None, the function returns the table that is not saved.
    :return: Returns a dataframe including with the appropriate format for upload on Ecotaxa. See instructions at https://ecotaxa.obs-vlfr.fr/gui/prj/
    """
    if instrument.lower()=='cytosense':
        df_ecotaxa_object=pd.concat([pd.DataFrame(
            {'img_file_name': ['[t]'] + list(df.img_file_name.values),
             'object_id': ['[t]'] + list(df.img_file_name.astype(str).str.replace('.png','').values),
             'object_lat': ['[f]'] + [cfg_metadata['latitude']] * len(df),
             'object_lon': ['[f]'] + [cfg_metadata['longitude']] * len(df),
             'object_date': ['[t]'] + [df.Sample.astype(str).values[0].split('_')[-2].replace('-','')] * len(df),
             'object_time': ['[t]'] + [df.Sample.astype(str).values[0].split('_')[-1].replace('h','')+'00'] * len(df),
             'object_depth_min': ['[f]'] + [as_float_cytosense_depth(df.Sample.astype(str).values[0][:-17].split('_')[-1])/100] * len(df),
             'object_depth_max': ['[f]'] + [as_float_cytosense_depth(df.Sample.astype(str).values[0][:-17].split('_')[-1])/100] * len(df)
             }),pd.concat([pd.DataFrame(dict(zip(('object_'+pd.Series(dict_properties_types.keys())).values,list(dict_properties_types.values())))),df.rename(columns=dict(zip(list(df.columns),list('object_'+df.columns))))[list(('object_'+pd.Series(dict_properties_types.keys())).values)]],axis=0).drop(columns=['object_image_intensity','object_slice']).reset_index(drop=True)],axis=1).reset_index(drop=True)
        df_ecotaxa_sample=pd.concat([pd.concat([pd.DataFrame({'sample_id': ['[t]'] + list(df.Sample.astype(str).values)}),pd.concat([df_sample_lexplore_cytosense,*[df_sample_lexplore_cytosense.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True)],axis=1),pd.DataFrame({'sample_volume_analyzed_ml': ['[f]'] + list(df.sample_volume_analyzed_ml.values),'sample_volume_pumped_ml': ['[f]'] + list(df.sample_volume_pumped_ml.values),'sample_volume_fluid_imaged_ml': ['[f]'] + list(df.sample_volume_fluid_imaged_ml.values),'sample_duration_sec': ['[f]'] + list(df.sample_duration_sec.values)})],axis=1)
        df_ecotaxa_acquisition=pd.concat([pd.concat([df_acquisition_lexplore_cytosense,*[df_acquisition_lexplore_cytosense.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True),pd.DataFrame({'acq_flow_rate': ['[f]'] + list(df.sample_flow_rate.astype(float).values),'acq_trigger_channel': ['[t]'] + list(df.sample_trigger.astype(str).values),'acq_trigger_threshold_mv': ['[f]'] + list(df.sample_trigger_threshold_mv.astype(float).values)})],axis=1)
        df_ecotaxa_process=pd.concat([df_processing_lexplore_cytosense,*[df_processing_lexplore_cytosense.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True)
        df_ecotaxa=pd.concat([df_ecotaxa_object,df_ecotaxa_sample,df_ecotaxa_acquisition,df_ecotaxa_process],axis=1)

    elif instrument.lower()=='flowcam':
        df_ecotaxa_object=pd.concat([pd.DataFrame(
            {'img_file_name': ['[t]'] + list(df.img_file_name.values),
             'object_id': ['[t]'] + list(df.img_file_name.astype(str).str.replace('.png','').values),
             'object_lat': ['[f]'] + [cfg_metadata['latitude']] * len(df),
             'object_lon': ['[f]'] + [cfg_metadata['longitude']] * len(df),
             'object_date': ['[t]'] + [df.Sample.astype(str).values[0].split('_')[-2].replace('-','')] * len(df),
             'object_time': ['[t]'] + [cfg_metadata['wasam_sampling_time']] * len(df),
             'object_depth_min': ['[f]'] + [cfg_metadata['wasam_sampling_depth']] * len(df),
             'object_depth_max': ['[f]'] + [cfg_metadata['wasam_sampling_depth']] * len(df)
             }),pd.concat([pd.DataFrame(dict(zip(('object_'+pd.Series([key for key in dict_properties_types_colors.keys() if key in df.columns])).values,[value for key,value in dict_properties_types_colors.items() if key in df.columns]))),df.rename(columns=dict(zip([column for column in df.columns if column in dict_properties_types_colors.keys()],['object_'+column for column in df.columns if column in dict_properties_types_colors.keys()])))[['object_'+column for column in df.columns if column in dict_properties_types_colors.keys()]]],axis=0).drop(columns=['object_image_intensity','object_slice']).reset_index(drop=True)],axis=1).reset_index(drop=True)
        df_ecotaxa_sample=pd.concat([pd.concat([pd.DataFrame({'sample_id': ['[t]'] + list(df.Sample.astype(str).values)}),pd.concat([df_sample_lexplore_flowcam,*[df_sample_lexplore_flowcam.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True)],axis=1),pd.DataFrame({'sample_volume_analyzed_ml': ['[f]'] + list(df.sample_volume_analyzed_ml.values),'sample_volume_pumped_ml': ['[f]'] + list(df.sample_volume_pumped_ml.values),'sample_volume_fluid_imaged_ml': ['[f]'] + list(df.sample_volume_fluid_imaged_ml.values),'sample_duration_sec': ['[t]'] + list(df.sample_duration_sec.values)})],axis=1)
        df_ecotaxa_acquisition=pd.concat([pd.concat([df_acquisition_lexplore_flowcam_10x,*[df_acquisition_lexplore_flowcam_10x.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True),pd.DataFrame({'acq_flow_rate': ['[f]'] + list(df.sample_flow_rate.astype(float).values)})],axis=1)
        df_ecotaxa_process=pd.concat([df_processing_lexplore_flowcam_10x,*[df_processing_lexplore_flowcam_10x.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True)
        df_ecotaxa=pd.concat([df_ecotaxa_object,df_ecotaxa_sample,df_ecotaxa_acquisition,df_ecotaxa_process],axis=1)

    else:
        print('Instrument not supported, please specify either "CytoSense" or "FlowCam". Quitting')
    if len(path_to_storage):
        df_ecotaxa.to_csv(path_to_storage, sep='\t',index=False)
    return df_ecotaxa
