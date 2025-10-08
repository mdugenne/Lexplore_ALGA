## Objective: This script performs a set of data processing steps on the short runs acquired with the CytoSense
import pandas as pd
import plotnine.scales
from plotnine import *
from plotnine import *
from pathlib import Path
from scripts.funcs_image_processing import cfg_metadata, nbss_estimates, path_to_git, theme_paper
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

## Workflow starts here:
path_to_export=Path(path_to_git /'data' / 'datafiles' / 'cytosense'  ).expanduser()
path_to_listmodes=list(path_to_export.rglob('*_Listmode.csv'))
df_listmode=pd.concat(map(lambda file: pd.read_csv(file,sep=r',',engine='python',encoding='utf-8').assign(Set=file.name.split('_',3)[-1].replace('_Listmode.csv',''),Date=file.name.split(' ',1)[1].split('_',1)[0])if (file.exists() & file.stat().st_size>0) else pd.DataFrame({'Set':file.name.split('_',3)[-1].replace('_Listmode.csv',''),'Date':file.name.split(' ',1)[1].split('_',1)[0]},index=[0]),path_to_listmodes)).reset_index(drop=True)
plt.hexbin(x=np.log10(df_listmode['FWS Total']), y=np.log10(df_listmode['FL Red Total']),gridsize=(500,500),cmap='twilight')
plt.show()
plt.savefig(fname='{}/figures/initial_test/Cytosense_cytogram.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')
df_listmode=pd.concat(map(lambda file: pd.read_csv(file,sep=r',',engine='python',encoding='utf-8').assign(Set=file.name.split('_',3)[-1].replace('_Listmode.csv',''),Date=file.name.split(' ',1)[1].split('_',1)[0])if (file.exists() & file.stat().st_size>0) else pd.DataFrame({'Set':file.name.split('_',3)[-1].replace('_Listmode.csv',''),'Date':file.name.split(' ',1)[1].split('_',1)[0]},index=[0]),[path for path in path_to_listmodes if 'lowFWS_lowFLR' in path.name])).reset_index(drop=True)
slope, intercept=0.4603280225470451, -1.0114195237533092
df_listmode['ECD']=(10**intercept)*(df_listmode['FWS Total']**slope)
df_listmode['datetime']=pd.to_datetime(df_listmode.Date, format='%Y-%m-%d %Hh%M')
df_listmode=df_listmode.dropna().reset_index(drop=True)
df_listmode.datetime.unique()
df_sub=df_listmode#[(df_listmode.datetime>np.datetime64("2025-06-01")) & (df_listmode.datetime<np.datetime64("2025-07-01"))]
df_sub=df_sub.assign(size_bin=lambda x: pd.cut(x.ECD,bins=2**np.arange(-2,10,1/16),labels=2**np.arange(-2,10,1/16)[:-1]+np.diff(2**np.arange(-2,10,1/16))/2)).astype({'size_bin':float}).groupby('datetime').size_bin.value_counts(normalize=True,sort=False).reset_index()
plot=(ggplot(df_sub)+ #[(df_sub.datetime>np.datetime64("2024-12-17")) & (df_sub.datetime<np.datetime64("2024-12-19"))]
geom_tile(mapping=aes(x='datetime.astype(str)',y='size_bin',fill='proportion'),stat='identity')+
 guides(colour=None, fill=None) +
 scale_fill_gradientn(colors=colors)+scale_y_log10()+
#scale_x_datetime(date_labels='%H:%M')+
 theme_paper).draw(show=False)
plot.show()

plt.hexbin(x=df_sub.datetime, y=(df_sub['ECD']),reduce_C_function=np.sum,cmap='twilight',yscale='log')
plt.show()
cmap = plt.cm.get_cmap('twilight', 100)
colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(100)]
plot=(ggplot(df_sub.query('ECD>7'))+
stat_density_2d(mapping=aes(x='datetime',y='(ECD)',fill=after_stat('density')),geom='tile',contour=False)+
 guides(colour=None, fill=None) +
 scale_fill_gradientn(colors=colors)+scale_y_log10()+
scale_x_datetime(date_labels='%H:%M')+
 theme_paper).draw(show=False)
plot.show()
plot.savefig(fname='{}/figures/Initial_test/dynamics_ECD_cytosense.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')
plot=(ggplot(df_listmode.groupby('datetime').ECD.agg(['mean','std']).reset_index())+
geom_pointrange(mapping=aes(x='datetime',y='mean',ymax='mean+std',ymin='mean-std'))+
 guides(colour=None, fill=None) +
scale_x_datetime(date_breaks='1 days',date_labels='%H:%M',limits=[np.datetime64('2025-05-01'),np.datetime64('2025-06-01')])+
 theme_paper).draw(show=False)
plot.show()

path_to_info=list(path_to_export.rglob('*_Info.txt'))
df_volume = pd.concat(map(lambda file: pd.read_table(file, engine='python', encoding='utf-8', sep=r'\t',names=['Variable']).assign(Value=lambda x: x.Variable.astype(str).str.split('\:|\>').str[1:].str.join('').str.strip(),Variable=lambda x: x.Variable.str.split('\:|\>').str[0]).set_index('Variable').T.rename( columns={'Volume (μL)': 'Volume_analyzed', 'Measurement duration': 'Measurement_duration'}).assign(Date=file.name.split(' ',1)[1].split('_',1)[0])[['Date','Volume_analyzed']],path_to_info))
path_to_summary=list(path_to_export.rglob('set_statistics*'))
df_summary = pd.concat(map(lambda file: pd.read_table(file, engine='python', encoding='utf-8', sep=r','),path_to_summary))
df_summary=df_summary.assign(Date=df_summary['Filename'].str.replace('.cyz','').str.split(' ').str[1:].str.join(' '))
df_summary['datetime']=pd.to_datetime(df_summary.Date,format='%Y-%m-%d %Hh%M')
df_summary['concentration']=df_summary['Concentration [n/μl]']*1000



plot = (ggplot(df_summary[df_summary['Set'].isin(['HighFWS_lowFLR','lowFWS_lowFLR'])],aes(x='datetime', y='concentration',group='Set',colour='Set')) +
        geom_point(mapping=aes(x='datetime', y='concentration',group='Set',colour='Set'), alpha=1)+
        geom_smooth(mapping=aes(x='datetime', y='concentration', group='Set', colour='Set'), se=False,method='loess',span=0.1) +
        labs(x='',y='Surface concentration (particles mL$^{-1}$)', title='',colour='') +
        scale_y_log10()+
        plotnine.scales.scale_colour_manual(values={'HighFWS_lowFLR':'#{:02x}{:02x}{:02x}'.format(145,111,111),'lowFWS_lowFLR':'#{:02x}{:02x}{:02x}'.format(164,93,111)})+
        #scale_y_log10(breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+04)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        #scale_x_log10( breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e+00)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e+00)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(), labels=lambda l: [v if ((v / (10 ** np.floor(np.log10(v)))) == 1) else '' for v in l]) +
        scale_x_date(date_breaks='1 month', date_labels='%b', limits=[datetime.date(2025, 1, 1), datetime.date(2025, 6, 30)]) +
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)
plot.set_size_inches(12, 4.5)
plot.show()
plot.savefig(fname='{}/figures/Initial_test/dynamics_cytosense.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')


df_nbss, df_nbss_boot=nbss_estimates(df=pd.merge(df_listmode,df_volume.assign(volume=lambda x: x.Volume_analyzed.astype(float)*1e-03)[['volume','Date']].drop_duplicates(),how='left',on=['Date']).assign(area=lambda x: (x['FWS Length']**2)*np.pi/4), pixel_size=1, grouping_factor=['Date'])
df_nbss_images, df_nbss_boot_images=nbss_estimates(df=df_ecotaxa.assign(volume=lambda x: x.sample_volume_fluid_imaged_ml.astype(float),area=lambda x: x.object_area), pixel_size=1, grouping_factor=['File'])
plot = (ggplot(df_nbss) +
        geom_point(mapping=aes(x='size_class_mid', y='NBSS',group='Group_index',colour='File'), alpha=1)+
        geom_point(data=df_nbss_images,mapping=aes(x='size_class_mid', y='NBSS', group='Group_index'), alpha=1) +
        labs(x='Equivalent circular diameter ($\mu$m)',y='Normalized Biovolume Size Spectra ($\mu$m$^{3}$ mL$^{-1}$ $\mu$m$^{-3}$)', title='',colour='') +
        scale_y_log10(breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+04)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        scale_x_log10( breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e+00)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e+00)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(), labels=lambda l: [v if ((v / (10 ** np.floor(np.log10(v)))) == 1) else '' for v in l]) +
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)
plot.show()