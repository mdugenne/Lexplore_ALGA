## Objective: This script generates summary data and figures from the database available at https://si-ola.inrae.fr/si_lacs/index.jsf#
# Load modules and functions required for image processing

try:
    from funcs_image_processing import *

except:
    from scripts.funcs_image_processing import *


import seaborn as sns
palette_oxygen=list(reversed(sns.color_palette("inferno",15).as_hex()))
palette_temp=list((sns.color_palette("BuPu",15).as_hex()))#colorspace.diverging_hcl(name="Purple-Blue").colors()
palette_chl=list((sns.color_palette("GnBu",15).as_hex()))#colorspace.diverging_hcl(name="GnBu").colors()
palette_bbp=list(reversed(sns.color_palette("RdBu",15).as_hex()))#colorspace.diverging_hcl(name="GnBu").colors()


# Data processing modules

from functools import reduce
from funcy import join_with
matplotlib.use('Qt5Agg')

dict_color={'Chl':'#{:02x}{:02x}{:02x}'.format(0,128,0),'PhycoCyanine':'#{:02x}{:02x}{:02x}'.format(200,55,55),'Orthophosphates':'#{:02x}{:02x}{:02x}'.format(255,153,85),'Ammonium':'#{:02x}{:02x}{:02x}'.format(108,83,83),'Temperature':'#{:02x}{:02x}{:02x}'.format(55,113,200)}

#Workflow starts here:
## Load datafiles downloaded from the datalakes portal
path_datafiles=natsorted(list(Path(path_to_git / 'data' / 'datafiles' / 'ola').expanduser().rglob('*ola_*.csv')))

df_env=pd.read_csv(path_datafiles[0],sep=';',decimal=',',encoding='latin-1',engine='python')
df_env.columns=['project', 'site', 'plateform','sampling_date', 'sampling_gear', 'measurement', 'depth_min', 'depth_max','depth', 'Temperature', 'Azote_total', 'Azote_organique_particulaire','Azote_Nitrates', 'Nitrates','Azote_Ammonium', 'Ammonium','Silice', 'Phosphore_Total','Phosphore_Particulaire', 'Phosphore_Orthophosphates','Orthophosphates']
df_env['datetime']=pd.to_datetime(df_env['sampling_date'],format='%d/%m/%Y')
df_env['season']=df_env.datetime.map(dict(map(lambda date: (date,season(date,'north')),df_env['datetime'].unique())))
df_env['Season']=pd.Categorical(df_env.season,["Winter","Spring","Summer","Fall"],ordered=True)
df_env['month']=df_env.datetime.dt.strftime('%m')

for variable, legend in {'Temperature':r'$\text{Temperature (}^{\degree})$C',
                         'Nitrates': r'$\text{NO}_{3-} \text{ concentration (mg L}^{-1})$',
                         'Orthophosphates': r'$\text{PO}_{4}^{3-} \text{ concentration (mg L}^{-1})$',
                         'Silice': r'$\text{SiO}_{2} \text{ concentration (mg L}^{-1}$)'}.items():
    # Long-term environmental timeseries
    plot = (ggplot(df_env.query('depth_min<10').dropna(subset=[variable])) +
            geom_line(mapping=aes(x='datetime', y=variable)) +
            labs(y=legend,x='', title='', colour='') +
            guides(colour=None, fill=None) +
            theme_paper+theme(text=element_text(size=22))).draw(show=False)
    plot.set_size_inches(12, 4.5)
    plot.show()
    plot.savefig(fname='{}/figures/ola/ts_all_{}.pdf'.format(str(path_to_git),variable), dpi=300, bbox_inches='tight')
    # Climatology
    plot = (ggplot(df_env.query('depth_min<10').dropna(subset=[variable])) +
            stat_summary(mapping=aes(x='month', y=variable,group='month'),geom='pointrange') +
            labs(y=legend, x='', title='', colour='') +
            guides(colour=None, fill=None) +
            theme_paper + theme(text=element_text(size=22))).draw(show=False)
    plot.set_size_inches(4, 4.5)
    plot.show()
    plot.savefig(fname='{}/figures/ola/climatology_{}.pdf'.format(str(path_to_git), variable), dpi=300, bbox_inches='tight')

df_phyto=pd.read_csv(path_datafiles[1],sep=';',decimal=',',encoding='latin-1',engine='python')
df_phyto.columns=['project', 'site', 'plateform','sampling_date', 'sampling_gear', 'measurement', 'depth_min', 'depth_max','technician', 'volume', 'count_surface', 'taxa','biovolume_concentration']
df_phyto['datetime']=pd.to_datetime(df_phyto['sampling_date'],format='%d/%m/%Y')
df_phyto['season']=df_phyto.datetime.map(dict(map(lambda date: (date,season(date,'north')),df_phyto['datetime'].unique())))
df_phyto['Season']=pd.Categorical(df_phyto.season,["Winter","Spring","Summer","Fall"],ordered=True)
df_phyto['month']=df_phyto.datetime.dt.strftime('%m')
# Generate lookup taxonomy table
#df_taxonomy=pd.concat(map(lambda hierarchy: annotation_in_WORMS(hierarchy),df_phyto.taxa.str.replace(' sp.','').unique())).reset_index(drop=True)
df_taxonomy=pd.DataFrame()
for hierarchy in df_phyto.taxa.str.replace(' sp.','').unique()[(np.where(df_phyto.taxa.str.replace(' sp.','').unique()==hierarchy)[0][0]+1):]:
    df_taxonomy=pd.concat([df_taxonomy,annotation_in_WORMS(hierarchy)],axis=0).reset_index(drop=True)
