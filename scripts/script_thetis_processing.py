## Objective: This script process the level 2 thetis netcdf datafiles downloaded on  https://www.datalakes-eawag.ch/data (product LeXPLORE Thetis Vertical Profiler Depth Time Grid)

import warnings
warnings.filterwarnings(action='ignore')

# Path and File processing modules
import os
import scipy as sp
from pathlib import Path
path_to_git=Path('~/GIT/Lexplore_ALGA').expanduser()
import shutil # zip folder
import re

# Module for netcdf and plots:
import xarray as xr # conda install -c conda-forge xarray dask netCDF4 bottleneck
import netCDF4 as nc # conda install -c conda-forge netCDF4
from dateutil.relativedelta import relativedelta
import itertools
from natsort import natsorted
import scipy
import plotly.express as px # Use conda install -c plotly plotly=5.24.1
import plotly.graph_objects as go
import seaborn as sns
palette_oxygen=list(reversed(sns.color_palette("inferno",15).as_hex()))
palette_temp=list((sns.color_palette("BuPu",15).as_hex()))#colorspace.diverging_hcl(name="Purple-Blue").colors()
palette_chl=list((sns.color_palette("GnBu",15).as_hex()))#colorspace.diverging_hcl(name="GnBu").colors()
palette_bbp=list(reversed(sns.color_palette("RdBu",15).as_hex()))#colorspace.diverging_hcl(name="GnBu").colors()

# Load modules and functions required for image processing

try:
    from funcs_image_processing import *

except:
    from scripts.funcs_image_processing import *

# Data processing modules
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm
from functools import reduce
from funcy import join_with
matplotlib.use('Qt5Agg')

dict_color={'Chl2':'#{:02x}{:02x}{:02x}'.format(0,128,0),'PhycoCy':'#{:02x}{:02x}{:02x}'.format(200,55,55),'PhycoEr':'#{:02x}{:02x}{:02x}'.format(255,153,85),'bb700':'#{:02x}{:02x}{:02x}'.format(108,83,83),'Temp':'#{:02x}{:02x}{:02x}'.format(55,113,200)}

#Workflow starts here:
## Load netcdf files after uncompressing the zip directory downloaded from the datalakes portal
path_datafiles=natsorted(list(Path(path_to_git / 'data' / 'datafiles' / 'thetis').expanduser().glob('*depthtimegrid*')))

for file in path_datafiles:

    if not Path(file.parent / file.name.replace(' ','_').replace('.zip','')).expanduser().exists():
        try:
            shutil.unpack_archive(file, file.parent / 'thetis_datalakes_download') # Unzip export file. Attention repository name needs to be changed to read nc files. Perhaps the utf-8 characters aren't supported
        except:
            print(r'Error uncompressing folder {}.\n Please check and re-export if needed'.format(str(file)))

path_datafiles=natsorted(list(Path(path_to_git / 'data' / 'datafiles' / 'thetis').expanduser().rglob('*.nc')))
# Identify monthly datafiles and generate monthly plots and reduce the memory load
dict_datafiles=join_with(list,[{file.stem.rsplit('_',2)[1][0:4]:file} for file in path_datafiles])
df_all=pd.DataFrame()
for date,datafiles in dict_datafiles.items():
    df=pd.concat(map(lambda path:xr.open_dataset(r'{}'.format(str(path))).to_dataframe().reset_index(),datafiles)).reset_index(drop=True)
    df=df.assign(season=lambda x: x.time.dt.strftime("%Y-%m-%dT%H:%M:%S").map(dict(map(lambda datetime: (format(datetime,"%Y-%m-%dT%H:%M:%S"),season(datetime,'north')),df.time.unique()))))
    ## Filter data outside temporal range
    df=df[df.time<pd.to_datetime('{}01'.format(date[0:4]+str(int(date[5:6])+1).zfill(2)),format='%Y%m%d')].reset_index(drop=True)
    df_all = pd.DataFrame()
    df_all=df.copy()#pd.concat([df_all,df],axis=0).reset_index(drop=True)
    df_all=df_all.assign(Depth=lambda x: pd.cut(x.Press,np.arange(0,200),right=False,labels=np.arange(0,200-1)+0.5).astype(str))
    df_all['datetime']=df_all.time.dt.floor('1h')
    df_all['time']=df_all.datetime.dt.hour
    df_summary=df_all.query('time==12').astype({'season':str}).groupby(['season','Depth'])[['Temp','PhycoEr','PhycoCy','Chl','Chl2']].agg(['mean', 'std']).reset_index()
    df_summary.Depth=df_summary.Depth.astype(float)
    df_summary=df_summary.sort_values(['season','Depth']).reset_index(drop=True)
    df_summary.columns=df_summary.columns.to_flat_index().str.join('_')
    df_summary['Season']=pd.Categorical(df_summary.season_,["Winter","Spring","Summer","Fall"],ordered=True)
    for variable, legend in {'bb700':r'$\text{Backscattering coefficient (m}^{-1})$)',
                             'Temp':r'$\text{Temperature (}^{\degree})$C',
                             'PhycoEr': r'$\text{Phycoerythrin concentration (mg m}^{-3})$',
                             'PhycoCy': r'$\text{Phycocyanin concentration (mg m}^{-3})$',
                             'Chl2': r'$\text{Chla concentration (mg m}^{-3}$)'}.items():

        plot = (ggplot(df_summary) +#(ggplot(df_all.assign(y=lambda x: np.where(x[variable]<0,pd.NA,x[variable]))) +
                facet_wrap('~Season',nrow=1,scales='free')+
                #stat_summary(mapping=aes(x='Press', y='y'),geom='pointrange', alpha=1) +
                geom_pointrange(fill=dict_color[variable],color='gray',mapping=aes(x='Depth_', y='{}_mean '.format(variable),ymin='({}_mean ) - np.sqrt({}_std)'.format(variable,variable),ymax='({}_mean ) + np.sqrt({}_std)'.format(variable,variable)),size=0.1)+
                geom_point(fill=dict_color[variable],mapping=aes(x='Depth_', y='{}_mean '.format(variable))) +
                coord_flip()+
                labs(y=legend,x='Depth (m)', title='', colour='') +
                scale_x_reverse(limits=[50,0]) +
                #scale_y_log10(labels=lambda labels: [10**(np.round(np.log10(label-1))) for label in labels]) +
                guides(colour=None, fill=None) +
                theme_paper).draw(show=False)
        plot.set_size_inches(8, 2.5)
        plot.show()
        plot.savefig(fname='{}/figures/datalakes/Hydronaut/profile_season_{}.pdf'.format(str(path_to_git),variable), dpi=300, bbox_inches='tight')

    for variable,legend in {'bb700':r'$\text{Backscattering coefficient (m}^{-1}$)','temp':r'$\text{Temperature (}^{\degree}$C)','chla':r'$\text{Chla concentration (mg m}^{-3}$)'}.items():
        ## Plot and save
        fig = go.Figure(data=go.Contour(x=df.time,y=df.depth, z=df[variable], colorbar=dict(orientation='v',tickangle=90,ticklabelposition='inside',x=1,titleside='right' ),contours=dict(size=2,coloring='heatmap'),line=dict(color='rgba(250,250,250 ,0)'),colorscale=['#ffffff']*5+palette_bbp))
        # reverse y-axis for depth
        fig.update_yaxes(autorange="reversed")
        # Add range slider
        fig.update_layout(  annotations=[ dict( text=legend, font_size=16,font_family='sans-serif',font_color='black', textangle=90,showarrow=False,xref="paper",yref="paper", x=1.07,y=0.5 )],xaxis=dict(rangeselector=dict( buttons=list([ dict(count=1, label="1m",step="month",stepmode="backward"), dict(step="all") ])),rangeslider=dict( visible=True),type="date"))
        path_to_plot=path_to_git / 'figures' / 'datalakes' / 'Thetis'
        path_to_plot.mkdir(parents=True, exist_ok=True)
        fig.write_html(path_to_plot/'Thetis_profile_{}_{}'.format(variable,date[0:4]+'_'+date[5:6].zfill(2)))#.show()


