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


# Size spectrum processing
from scipy.stats import poisson
from scipy import stats
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
    x_melt = pd.merge(x_melt,df_summary , how='left', on=grouping_factor+['Group_index','volume','size_class'])

    return x_melt,df_nbss_boot