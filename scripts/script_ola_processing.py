## Objective: This script generates summary data and figures from the database available at https://si-ola.inrae.fr/si_lacs/index.jsf#
# Load modules and functions required for image processing
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
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

dict_color={'Chl':'#{:02x}{:02x}{:02x}'.format(0,128,0),'PhycoCyanine':'#{:02x}{:02x}{:02x}'.format(200,55,55),'Orthophosphates':'#{:02x}{:02x}{:02x}'.format(255,153,85),'Ammonium':'#{:02x}{:02x}{:02x}'.format(108,83,83),'Temperature':'#{:02x}{:02x}{:02x}'.format(55,113,200)}

#Workflow starts here:
## Load datafiles downloaded from the datalakes portal
path_datafiles=natsorted(list(Path(path_to_git / 'data' / 'datafiles' / 'ola').expanduser().rglob('*ola_*.csv')))

df_env=pd.read_csv(path_datafiles[1],sep=';',decimal=',',encoding='latin-1',engine='python')
df_env.columns=['project', 'site', 'plateform','sampling_date', 'sampling_gear', 'measurement', 'depth_min', 'depth_max','depth', 'Temperature', 'Azote_total', 'Azote_organique_particulaire','Azote_Nitrates', 'Nitrates','Azote_Ammonium', 'Ammonium','Azote_Nitrites', 'Nitrites','Silice','Carbon_Organic_Particular','Oxygen', 'Phosphore_Total','Phosphore_Particulaire', 'Phosphore_Orthophosphates','Orthophosphates']#['project', 'site', 'plateform','sampling_date', 'sampling_gear', 'measurement', 'depth_min', 'depth_max','depth', 'Temperature', 'Azote_total', 'Azote_organique_particulaire','Azote_Nitrates', 'Nitrates','Azote_Ammonium', 'Ammonium','Silice', 'Phosphore_Total','Phosphore_Particulaire', 'Phosphore_Orthophosphates','Orthophosphates']
df_env['datetime']=pd.to_datetime(df_env['sampling_date'],format='%d/%m/%Y')
df_env['season']=df_env.datetime.map(dict(map(lambda date: (date,season(date,'north')),df_env['datetime'].unique())))
df_env['Season']=pd.Categorical(df_env.season,["Winter","Spring","Summer","Fall"],ordered=True)
df_env['month']=df_env.datetime.dt.strftime('%m')

df_env_summary=df_env.query('depth_max==5').assign(year=lambda x: x.datetime.dt.strftime('%Y')).groupby('year').Phosphore_Total.mean().reset_index()
df_env_summary=df_env_summary.assign(trophic_state=lambda x: pd.cut(1000*x.Phosphore_Total,[0,10,20,100],labels=['oligotrophic','mesotrophic','eutrophic']))
periods=df_env_summary.dropna(subset=['trophic_state']).groupby('trophic_state').year.first()

for variable, legend in {'Temperature':r'$\text{Temperature (}^{\degree})$C',
                         'Nitrates': r'$\text{NO}_{3-} \text{ concentration (mg/L)$',
                         'Orthophosphates': r'$\text{PO}_{4}^{3-} \text{ concentration (mg/L)}$',
                         'Phosphore_Total': r'$\text{Total phosphorus} \text{ concentration (mg/L)}$',
                         'Silice': r'$\text{SiO}_{2} \text{ concentration (mg L}^{-1}$)'}.items():
    # Long-term environmental timeseries
    fig, axes = plt.subplots(1, 1,frameon=False)  # figsize=tuple(np.array(padded_image.shape)[0:2][::-1]*40/300),dpi=300

    plot = (ggplot(df_env.query('depth_min==5').dropna(subset=[variable])) +
            geom_line(mapping=aes(x='datetime', y=variable)) +
            scale_y_continuous(minor_breaks=4)+
            labs(y=legend,x='', title='', colour='') +
            guides(colour=None, fill=None) +
            theme(text=element_text(size=22))).draw(show=False)
    plot.set_size_inches(12, 4.5)
    plot.show()
    plot.savefig(fname='{}/figures/ola/ts_all_{}.pdf'.format(str(path_to_git),variable), dpi=300, bbox_inches='tight')
    # Climatology
    plot = (ggplot(df_env[df_env.datetime.dt.year>2020].query('(depth_min==5)').dropna(subset=[variable])) +
            stat_summary(mapping=aes(x='month', y=variable,group='month'),geom='pointrange') +
            labs(y=legend, x='Month', title='', colour='') +
            guides(colour=None, fill=None) +
            theme_paper + theme(text=element_text(size=22))).draw(show=False)
    plot.set_size_inches(6, 4.5)
    plot.show()
    plot.savefig(fname='{}/figures/ola/climatology_{}.pdf'.format(str(path_to_git), variable), dpi=300, bbox_inches='tight')
    # Depth profile
    plot = (ggplot(df_env[df_env.datetime.dt.year>2020].dropna(subset=[variable]).query('(month=="05")').groupby(['depth_min'])[variable].agg(['mean','std']).reset_index()) +
            geom_pointrange(mapping=aes(x='depth_min', y='mean',ymax='mean+std',ymin='mean-std')) +
            coord_flip()+scale_x_reverse()+scale_y_log10()+
            labs(y=legend, x='Depth (m)', title='', colour='') +
            guides(colour=None, fill=None) +
            theme_paper + theme(text=element_text(size=22))).draw(show=False)
    plot.set_size_inches(6, 4.5)
    plot.show()
    plot.savefig(fname='{}/figures/ola/profile_{}.pdf'.format(str(path_to_git), variable), dpi=300, bbox_inches='tight')

df_phyto=pd.read_csv(path_datafiles[3],sep=';',decimal=',',encoding='latin-1',engine='python')
df_phyto.columns=['project', 'site', 'plateform','sampling_date', 'sampling_gear', 'measurement', 'depth_min', 'depth_max','technician', 'volume', 'count_surface', 'taxa','biovolume_concentration']
df_phyto['datetime']=pd.to_datetime(df_phyto['sampling_date'],format='%d/%m/%Y')
df_phyto['season']=df_phyto.datetime.map(dict(map(lambda sampling_date: (sampling_date,season(pd.to_datetime(sampling_date),'north')),df_phyto['datetime'].unique())))
df_phyto['Season']=pd.Categorical(df_phyto.season,["Winter","Spring","Summer","Fall"],ordered=True)
df_phyto['month']=df_phyto.datetime.dt.strftime('%m')
df_phyto['year']=df_phyto.datetime.dt.strftime('%Y').astype(float)
df_phyto.taxa=df_phyto.taxa.str.replace('Cellule de ','')
df_phyto.taxa=df_phyto.taxa.str.replace('cee','ceae')
df_phyto.taxa=df_phyto.taxa.str.replace('cées','ceae')
natsorted(df_phyto.taxa.unique())
# Using Rimet et al. 2020 density in freshwater as 1 kilogram per liter
df_phyto['biomass_ug_L']=df_phyto.biovolume_concentration*1e-03


# Generate lookup taxonomy table
if not Path(path_datafiles[1].parent / 'taxonomy_lookup_table.csv').exists():
    df_taxonomy=pd.concat(map(lambda hierarchy: annotation_in_WORMS(hierarchy),df_phyto.taxa.str.replace(' sp.','').unique())).reset_index(drop=True)
    df_taxonomy=pd.read_csv(path_datafiles[1].parent / 'taxonomy_lookup_table.csv')
    while len(df_taxonomy)!=len(df_phyto.taxa.str.split(' ').str[0].str.replace('cées','ceae').unique()):
        ind=np.arange(0,len(df_phyto.taxa.str.split(' ').str[0].str.replace('cées','ceae').unique())+1) if len(df_taxonomy)==0 else np.arange(np.where(df_phyto.taxa.str.split(' ').str[0].unique()==hierarchy)[0][0]+1,len(df_phyto.taxa.str.split(' ').str[0].unique())+1)
        ind=np.where(pd.Series(df_phyto.taxa.str.split(' ').str[0].str.replace('cées','ceae').unique()).isin(df_taxonomy.EcoTaxa_hierarchy.tolist())==False)[0]
        ind=np.where(pd.Series(df_phyto.taxa.str.split(' ').str[0].str.replace('cées','ceae').unique()).isin(df_taxonomy.astype({'Full_hierarchy':str}).query('Full_hierarchy!="nan"').EcoTaxa_hierarchy.tolist())==False)[0]
        for hierarchy in natsorted(df_phyto.taxa.str.split(' ').str[0].str.replace('cées','ceae').unique()[ind]):
            print(hierarchy)
            try:
                df_taxonomy=pd.concat([df_taxonomy,annotation_in_WORMS(hierarchy)],axis=0).reset_index(drop=True)
            except:
                pass
    df_taxonomy.astype({'Full_hierarchy':str}).query('Full_hierarchy!="nan"').to_csv(path_datafiles[1].parent / 'taxonomy_lookup_table.csv',index=False)
else:
    df_taxonomy=pd.read_csv(path_datafiles[3].parent / 'taxonomy_lookup_table.csv')
df_phyto=pd.merge(df_phyto.assign(EcoTaxa_hierarchy=lambda x: x.taxa.str.split(' ').str[0]),df_taxonomy,how='left',on=['EcoTaxa_hierarchy'])
df_test=df_phyto.loc[df_phyto.Class.isna(),df_taxonomy.columns].drop_duplicates()

# Size surface to biovolume ratio and size classes
df_phyto=pd.read_csv(path_datafiles[3],sep=';',decimal=',',encoding='latin-1',engine='python')
df_phyto.columns=['project', 'site', 'plateform','sampling_date', 'sampling_gear', 'measurement', 'depth_min', 'depth_max','technician', 'volume', 'count_surface', 'taxa','biovolume_concentration']
df_phyto['datetime']=pd.to_datetime(df_phyto['sampling_date'],format='%d/%m/%Y')
df_phyto['season']=df_phyto.datetime.map(dict(map(lambda sampling_date: (sampling_date,season(pd.to_datetime(sampling_date),'north')),df_phyto['datetime'].unique())))
df_phyto['Season']=pd.Categorical(df_phyto.season,["Winter","Spring","Summer","Fall"],ordered=True)
df_phyto['month']=df_phyto.datetime.dt.strftime('%m')
df_phyto['year']=df_phyto.datetime.dt.strftime('%Y').astype(float)
df_phyto=df_phyto[((df_phyto.year==2015) & (df_phyto.depth_max==20))==False]

df_properties=pd.read_csv(path_datafiles[3].parent / "propriete_taxon__1756362700932-308701.csv",encoding='latin-1',engine='python',sep=';',index_col=False)
old_columns=df_properties.columns
df_properties.columns=['taxa','description_author','description_year','reference','Bourelly_class','Sandre_code','Cemagref_code','INRA_code','Counting_unit','Biovolume_error','unit_biovolume','unit_length','unit_width','unit_thickness','cell_geometric_shape','cell_multiplication_biovolume','cell_division_surface','cell_surface','cell_biovolume','cell_number','cumulative_surface','cumulative_biovolume','mucilage_length','mucilage_width','mucilage_thickness','colony_geometric_shape','colony_multiplication_biovolume','colony_division_surface','colony_surface','colony_biovolume','notes','basionyme','Omnidia_code','INRA_code_letters','Rebecca_nomenclature','Rebecca_code','uncertainty_level','recent_basionyme','descriptoin_reference','comments']
df_phyto=pd.merge(df_phyto,df_properties[['taxa','unit_biovolume','cell_surface','cell_biovolume','colony_surface','colony_biovolume','cumulative_biovolume','cumulative_surface']],how='left',on='taxa')
df_phyto['ESD']=(df_phyto[['unit_biovolume','cell_biovolume','colony_biovolume']].max(axis=1)*6/np.pi)**(1/3)
df_phyto['size_class']=pd.cut(df_phyto['ESD'],[0.2,2,20,200,2000],labels=['pico','nano','micro','meso']).astype(str)
df_phyto['surface_to_biovolume']=df_phyto[['cumulative_surface','cell_surface','colony_surface']].max(axis=1)/df_phyto[['cumulative_biovolume','cell_biovolume','colony_biovolume']].max(axis=1)


df_phyto['cell_concentration']=df_phyto.biovolume_concentration/df_phyto[['unit_biovolume','cell_biovolume','colony_biovolume']].max(axis=1)
df_phyto=df_phyto.query('size_class.isin(["pico","nano","micro"])')
df_phyto['total_concentration']=df_phyto.astype(dict(zip(['depth_min','depth_max','year'],[str]*3))).groupby(['depth_min','depth_max','year']).cell_concentration.transform('sum')
df_phyto['relative_concentration']=df_phyto.cell_concentration/df_phyto.total_concentration
df_phyto.groupby(['depth_min','depth_max','year']).relative_concentration.sum()

df_summary=df_phyto.astype({'relative_concentration':float,'depth_min':str,'depth_max':str,'datetime':str,'year':str,'size_class':str}).groupby(['depth_min','depth_max','year','size_class']).relative_concentration.sum().reset_index().sort_values('year')
df_summary=((df_phyto.astype({'relative_concentration':float,'depth_min':str,'depth_max':str,'datetime':str,'year':str,'size_class':str}).groupby(['depth_min','depth_max','year','size_class']).size_class.value_counts())/df_phyto.astype({'relative_concentration':float,'depth_min':str,'depth_max':str,'datetime':str,'year':str,'size_class':str}).groupby(['depth_min','depth_max','year','size_class']).size_class.value_counts().groupby(['depth_min','depth_max','year']).sum()).reset_index().rename(columns={'count':'relative_concentration'})
df_test=df_phyto.loc[df_phyto.size_class=='pico',['taxa','colony_biovolume','cell_biovolume','cumulative_biovolume','ESD','size_class']].drop_duplicates()

plot=(ggplot(df_phyto)+
      geom_area(mapping=aes(x='year',y=after_stat('count'),group='size_class',fill='size_class'), position="fill")+
      scale_fill_manual(values={'micro':'#{:02x}{:02x}{:02x}'.format(55,55,200),'nano':'#{:02x}{:02x}{:02x}'.format(44,160,137),'pico':'#{:02x}{:02x}{:02x}'.format(170,212,0)})+
      labs(x='',y='Relative abundance',fill='')+theme_paper + theme(text=element_text(family='Times New Roman',colour='black',size=20))).draw(show=True)
plot.set_size_inches(10, 4.5)
plot=(ggplot(df_summary.query('size_class.isin(["pico","nano","micro"])').astype({'year':float}))+
      geom_area(mapping=aes(x='year',y='relative_concentration',group='size_class',fill='size_class'))+
      scale_fill_manual(values={'micro':'#{:02x}{:02x}{:02x}'.format(55,55,200),'nano':'#{:02x}{:02x}{:02x}'.format(44,160,137),'pico':'#{:02x}{:02x}{:02x}'.format(170,212,0)})+
      labs(x='',y='Relative abundance',fill='')+theme_paper + theme(text=element_text(family='Times New Roman',colour='black',size=20))).draw(show=True)
plot.set_size_inches(10, 4.5)
plot.savefig(fname='{}/figures/ola/ts_size_classes.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')

df_summary=df_phyto.astype({'relative_concentration':float,'depth_min':str,'depth_max':str,'year':str}).groupby(['depth_min','depth_max','year']).apply(lambda x: pd.Series({'weighted_surface_volume':sum(x.relative_concentration*x.surface_to_biovolume)/len(x)})).reset_index()
plot=(ggplot(df_summary.astype({'year':float}))+
      geom_line(mapping=aes(x='year',y='weighted_surface_volume'),linetype='dashdot',colour='red')+
      geom_point(mapping=aes(x='year',y='weighted_surface_volume'),colour='red')+
      labs(x='',y='Weighted Surface:Volume ratio',fill='')+theme_paper + theme(text=element_text(family='Times New Roman',colour='black',size=20))).draw(show=True)
plot.set_size_inches(10, 4.5)
plot.savefig(fname='{}/figures/ola/ts_size_classes.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')

# Volume-to-carbon conversion
# concentration in pg C per ml = 1e-03 ug per l
df_phyto['biomass_concentration']=np.where(df_phyto.Class=='Bacillariophyceae',((1e-03)/df_phyto.depth_max)*(10**-0.933)*(df_phyto.biovolume_concentration**0.881),((1e-03)/df_phyto.depth_max)*(10**-0.665)*(df_phyto.biovolume_concentration**0.939))
df_summary=df_phyto.groupby(['sampling_date','Class']).biomass_concentration.sum().reset_index().pivot_table(index=['sampling_date'],columns=['Class']).reset_index()
df_summary=df_phyto.pivot_table(values=['biomass_concentration'],columns=['Class'],index=['sampling_date', 'depth_min', 'depth_max'],aggfunc=np.sum).reset_index()
df_summary['total_biomass']=df_summary.biomass_concentration.sum(axis=1)
df_summary=df_summary.reset_index()
df_summary.columns=[col[-1] if col[-1]!='' else col[0] for col in df_summary.columns]
df_summary['sampling_datetime']=pd.to_datetime(df_summary.sampling_date,format='%d/%m/%Y')
df_summary=df_summary.sort_values('sampling_datetime').reset_index(drop=True)
taxa=['Cyanophyceae','Cryptophyceae','Bacillariophyceae','Chlorophyceae','Chrysophyceae','Zygnematophyceae']
palette_taxa=dict(zip(taxa,['#{:02x}{:02x}{:02x}'.format(111,145,138),'#{:02x}{:02x}{:02x}'.format(222,135,26),'#{:02x}{:02x}{:02x}'.format(170,136,0),'#{:02x}{:02x}{:02x}'.format(34,136,0),'#{:02x}{:02x}{:02x}'.format(128,102,0),'#{:02x}{:02x}{:02x}'.format(85,0,0)]))
variable='total_biomass'

#Class-specific climatologies

df_summary['periods']=pd.cut(df_summary.sampling_datetime,include_lowest=True,bins=[np.datetime64(periods['eutrophic']),np.datetime64(periods['mesotrophic']),np.datetime64(periods['oligotrophic']),np.datetime64('2026-01-01'),],labels=['Eutrophic','Mesotrophic','Oligotrophic']).astype(str)
plot=(ggplot(pd.concat([df_summary[['sampling_datetime']],(df_summary[taxa].T/np.nansum(df_summary[taxa],axis=1)).T],axis=1).assign(month=lambda x: x.sampling_datetime.dt.month).melt(id_vars=['month'],value_vars=taxa))+
      facet_wrap('~variable',ncol=2)+
      geom_boxplot(mapping=aes(x='month',y='value',group='month'))+labs(x='',y='')+scale_y_sqrt()+scale_x_continuous(breaks=np.arange(1,12,4))+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.set_size_inches(6, 10.5)
plot.show()
plot.savefig(fname='{}/figures/ola/climatology_{}.pdf'.format(str(path_to_git),'_'.join(taxa)), dpi=300, bbox_inches='tight')

id=0
# Top-5 genus bar plot and pie chart
df_summary=df_phyto.groupby(['sampling_date','Class','Genus']).biomass_concentration.sum().reset_index().groupby(['Class','Genus']).biomass_concentration.max().reset_index().groupby(['Class']).apply(lambda x: x.nlargest(10,'biomass_concentration'))
import plotly.express as px
import plotly.io as pio
import kaleido
pio.renderers.default = "browser"
taxa=['Cyanophyceae','Bacillariophyceae','Chlorophyceae','Chrysophyceae','Zygnematophyceae']

fig = px.pie(df_summary.query("Class.isin(['{}'])".format(taxa[4]))[["Genus","biomass_concentration"]],
             color_discrete_sequence=px.colors.sequential.amp, values='biomass_concentration', names='Genus',hole=.6)
fig.show()
fig.write_image('{}/figures/ola/pie_{}.pdf'.format(str(path_to_git),taxa[4]),engine='orca')

plot=(ggplot(df_summary.query('Class=="{}"'.format(taxa[id])).sort_values('biomass_concentration'))+
      theme(axis_text_x=element_text(rotation=45,hjust=1))+
      geom_bar(mapping=aes(x='factor(Genus,["{}"])'.format('","'.join(df_summary.query('Class=="{}"'.format(taxa[id])).Genus)),y='biomass_concentration'),stat='identity')+scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.show()
plot.savefig(fname='{}/figures/ola/top10_{}.pdf'.format(str(path_to_git),taxa[id]), dpi=300, bbox_inches='tight')
from colour import Color
plot=(ggplot(df_summary[['sampling_datetime','periods']+taxa].assign(month=lambda x: x.sampling_datetime.dt.month).astype({'month':str}).melt(id_vars=['month','periods'],value_vars=taxa[id]))+
      facet_wrap('~variable',ncol=2)+
      scale_fill_manual(dict(zip(['Eutrophic','Mesotrophic','Oligotrophic'],list(map(lambda col: col.hex(),Color(palette_taxa[taxa[id]] ).gradient(to=Color('#ffffffff'), steps=3))))))+
      scale_x_continuous(breaks=np.arange(1,12,2))+
      geom_boxplot(mapping=aes(x='month.astype(float)',y='value',group="month + '.' + periods",fill='periods'),position=position_dodge(0))+scale_y_sqrt()+theme_paper + theme(legend_position='bottom',text=element_text(size=22))).draw(show=False)
plot.show()
plot.savefig(fname='{}/figures/ola/climatology_{}.pdf'.format(str(path_to_git),taxa[id]), dpi=300, bbox_inches='tight')

#Entire time-series
plot=(ggplot(df_summary)+geom_line(mapping=aes(x='sampling_datetime',y=variable))+scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.set_size_inches(11.4, 4.5)
plot=(ggplot(df_summary)+geom_line(mappin
      geom_area(mapping=aes(x='sampling_datetime',y=taxa[0]),color='#{:02x}{:02x}{:02x}'.format(111,145,138),fill='#{:02x}{:02x}{:02x}'.format(111,145,138),alpha=0.5)+
    #  geom_area(mapping=aes(x='sampling_datetime',y=taxa[1]),color='#{:02x}{:02x}{:02x}'.format(222,135,26),fill='#{:02x}{:02x}{:02x}'.format(229,145,125),alpha=0.5)+
    #  geom_area(mapping=aes(x='sampling_datetime',y=taxa[2]),color='#{:02x}{:02x}{:02x}'.format(170,136,0),fill='#{:02x}{:02x}{:02x}'.format(170,136,0),alpha=0.5)+
    #  geom_area(mapping=aes(x='sampling_datetime',y=taxa[3]),color='#{:02x}{:02x}{:02x}'.format(34,136,0),fill='#{:02x}{:02x}{:02x}'.format(34,136,0),alpha=0.5)+
    #  geom_area(mapping=aes(x='sampling_datetime',y=taxa[4]),color='#{:02x}{:02x}{:02x}'.format(128,102,0),fill='#{:02x}{:02x}{:02x}'.format(128,102,0),alpha=0.5)+
    #  geom_area(mapping=aes(x='sampling_datetime',y=taxa[5]),color='#{:02x}{:02x}{:02x}'.format(222,135,135),fill='#{:02x}{:02x}{:02x}'.format(222,135,135),alpha=0.5)+
      scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot=(ggplot(df_summary.melt(id_vars=['sampling_datetime','total_biomass'],value_vars=taxa))+scale_fill_manual(palette_taxa)+
      geom_area(mapping=aes(x='sampling_datetime',y='value',fill='variable'),alpha=0.7)+
      geom_line(data=df_summary,mapping=aes(x='sampling_datetime',y=variable),size=2.3)+
      labs(x='',y='')+
      scale_x_datetime(limits=[np.datetime64('2000-01-01'),np.datetime64('2000-12-31')],date_breaks='3 months',date_labels='%b',date_minor_breaks='1 months')+
      scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.set_size_inches(6.4, 4.5)
plot.show()
plot.savefig(fname='{}/figures/ola/ts_2001_{}.pdf'.format(str(path_to_git), variable), dpi=300, bbox_inches='tight')

#Compare with flowcam ALGA data
# Plot time-series
save_directory = Path(cfg_metadata['flowcam_10x_context_file'].replace('acquisitions', 'ecotaxa')).expanduser().parent / 'sample_id'
df_nbss=pd.concat(map(lambda path_ecotaxa:(nbss_estimates(df=pd.read_csv(path_ecotaxa,sep='\t').drop(index=[0]).astype({'object_area':float,'sample_volume_fluid_imaged_ml':float}).rename(columns={'object_area':'area','sample_volume_fluid_imaged_ml':'volume'}), pixel_size=1, grouping_factor=['sample_id'])[0]).assign(instrument=lambda x:np.where(x.sample_id.str.contains('Flowcam_2mm'),'FlowCam Macro','FlowCam Micro')),natsorted(list(save_directory.parent.rglob('ecotaxa_table_*'))))).reset_index(drop=True).rename(columns={'sample_id':'Sample'})
df_nbss=pd.concat([df_nbss,pd.concat(map(lambda path_ecotaxa:(nbss_estimates(df=pd.read_csv(path_ecotaxa,sep='\t').drop(index=[0]).astype({'object_area':float,'sample_volume_fluid_imaged_ml':float}).rename(columns={'object_area':'area','sample_volume_fluid_imaged_ml':'volume'}), pixel_size=1, grouping_factor=['sample_id'])[0]).assign(instrument="CytoSense"),natsorted(list(Path(path_to_network / 'lexplore' / 'LEXPLORE' / 'ecotaxa' ).rglob('ecotaxa_table_*'))))).reset_index(drop=True).rename(columns={'sample_id':'Sample'})],axis=0)
# Transform NBSS in biomass cubic micrometers per ml = pgC per mL = 1-06 x 1+03
df_summary_nbss=df_nbss.query('instrument=="FlowCam Micro"').assign(biomass=lambda x: ((1e-03)*(10**-0.933)*(x.NBSS*x.range_size_bin)**0.881)/18).groupby('Sample').biomass.sum().reset_index().assign(sampling_date=lambda x:pd.to_datetime(x.Sample.str.split('_').str[4],format='%Y%m%d'))

plot = (ggplot(df_summary_nbss) +
        stat_summary(data=df_summary.assign(timestamp=pd.to_datetime('2025-'+df_summary.sampling_datetime.dt.strftime('%m-%d'))),mapping=aes(x='timestamp',y='total_biomass',group='timestamp'),geom='pointrange')+
        #geom_area(data=df_summary.assign(timestamp=pd.to_datetime('2025-' +df_summary.sampling_datetime.dt.strftime('%m-%d'))), mapping=aes(x='timestamp', y='total_biomass')) +
        #geom_bar(data=pd.concat([df_summary.sampling_datetime,df_summary[taxa].cumsum(axis=1)],axis=1).assign(timestamp=pd.to_datetime('2025-' + df_summary.sampling_datetime.dt.strftime('%m-%d'))).groupby(['timestamp'])['Zygnematophyceae'].mean().reset_index(), mapping=aes(x='timestamp', y='Zygnematophyceae'),stat='identity',fill=palette_taxa['Zygnematophyceae'],alpha=.6) +
        #geom_bar(data=pd.concat([df_summary.sampling_datetime,df_summary[taxa].cumsum(axis=1)],axis=1).assign( timestamp=pd.to_datetime('2025-' + df_summary.sampling_datetime.dt.strftime('%m-%d'))).groupby(['timestamp'])['Bacillariophyceae'].mean().reset_index(),  mapping=aes(x='timestamp', y='Bacillariophyceae'),stat='identity',fill=palette_taxa['Bacillariophyceae'],alpha=1) +
        #geom_bar(data=pd.concat([df_summary.sampling_datetime,df_summary[taxa].cumsum(axis=1)],axis=1).assign( timestamp=pd.to_datetime('2025-' + df_summary.sampling_datetime.dt.strftime('%m-%d'))).groupby(['timestamp'])['Chrysophyceae'].mean().reset_index(), mapping=aes(x='timestamp', y='Chrysophyceae'),stat='identity',fill=palette_taxa['Chrysophyceae'],alpha=1) +
        #geom_bar(data=pd.concat([df_summary.sampling_datetime,df_summary[taxa].cumsum(axis=1)],axis=1).assign( timestamp=pd.to_datetime('2025-' + df_summary.sampling_datetime.dt.strftime('%m-%d'))).groupby(['timestamp'])['Cyanophyceae'].mean().reset_index(), mapping=aes(x='timestamp', y='Cyanophyceae'),stat='identity',fill=palette_taxa['Cyanophyceae'],alpha=1) +
        geom_line(mapping=aes(x='sampling_date',y='biomass'))+
        stat_summary(data=df_env.query('depth_max==5').assign(timestamp=lambda x: pd.to_datetime('2025-'+x.datetime.dt.strftime('%m-%d'))),mapping=aes(x='timestamp',y='1000*Carbon_Organic_Particular'),colour='blue',geom='line')+ # mg/L
        labs(x='',y='Total biomass ($\mu$g L$^{-1}$)', title='',colour='') +
        scale_x_date(limits=[pd.to_datetime('2025-01-01'),pd.to_datetime('2025-12-15')])+
        scale_y_sqrt() +
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)
plot.set_size_inches(12, 4.5)
plot.show()
plot.savefig(fname='{}/figures/ola/ts_all_biomass_with_flowcam.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')

# Check mixotrophic taxa
df_mixo=pd.read_csv(path_to_git / 'data' / "NanoplanktonNutritionStrategiesDBLeNoachBeisner2024_fbae035.csv",encoding='latin-1').query('Final_Nutrition_Strategy=="Mixotroph"')
df_phyto['Total_biomass']=df_phyto.groupby(['datetime','depth_min','depth_max']).biomass_concentration.transform('sum')/df_phyto.depth_max
df_phyto['Relative_biomass']=df_phyto.biomass_concentration/df_phyto.depth_max/df_phyto.Total_biomass

df_phyto_mixo=df_phyto.query('EcoTaxa_hierarchy.isin(["{}"])'.format('","'.join(df_mixo.Genus.unique())))
df_summary=df_phyto_mixo.groupby(['datetime','depth_min','depth_max']).Relative_biomass.sum().reset_index()
plot=(ggplot(df_summary)+geom_line(mapping=aes(x='datetime',y='Relative_biomass'))+scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.set_size_inches(12, 4.5)
plot.show()
plot.savefig(fname='{}/figures/ola/ts_all_biomass_with_fmixo.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')



df_zoo=pd.read_csv(path_datafiles[-2],sep=';',decimal=',',encoding='latin-1',engine='python')
df_zoo.columns=['project', 'site', 'plateform','sampling_date', 'sampling_gear', 'measurement', 'depth_min', 'depth_max','technician', 'biovolume', 'taxa','stage','varaible','concentration']
df_zoo['datetime']=pd.to_datetime(df_zoo['sampling_date'],format='%d/%m/%Y')
df_zoo['season']=df_zoo.datetime.map(dict(map(lambda sampling_date: (sampling_date,season(pd.to_datetime(sampling_date),'north')),df_zoo['datetime'].unique())))
df_zoo['Season']=pd.Categorical(df_zoo.season,["Winter","Spring","Summer","Fall"],ordered=True)
df_zoo['month']=df_zoo.datetime.dt.strftime('%m')
# Generate lookup taxonomy table
df_zoo['taxa']=df_zoo.taxa.str.replace('ères|ère','era')
while len(df_taxonomy)+len(df_zoo.taxa.str.split(' ').str[0].str.unique())!=len(df_zoo.taxa.str.split(' ').str[0].str.unique()):
    ind=np.arange(0,len(df_zoo.taxa.str.split(' ').str[0].unique())+1) if len(df_taxonomy)==0 else np.arange(np.where(df_zoo.taxa.str.split(' ').str[0].unique()==hierarchy)[0][0]+1,len(df_zoo.taxa.str.split(' ').str[0].unique())+1)
    ind=np.where(pd.Series(df_zoo.taxa.str.split(' ').str[0].unique()).isin(df_taxonomy.EcoTaxa_hierarchy.tolist())==False)[0]
    ind=np.where(pd.Series(df_zoo.taxa.str.split(' ').str[0].unique()).isin(df_taxonomy.astype({'Full_hierarchy':str}).query('Full_hierarchy!="nan"').EcoTaxa_hierarchy.tolist())==False)[0]
    for hierarchy in natsorted(df_zoo.taxa.str.split(' ').str[0].unique()[ind]):
        print(hierarchy)
        try:
            df_taxonomy=pd.concat([df_taxonomy,annotation_in_WORMS(hierarchy)],axis=0).reset_index(drop=True)
        except:
            pass
df_taxonomy.astype({'Full_hierarchy':str}).query('Full_hierarchy!="nan"').to_csv(path_datafiles[1].parent / 'taxonomy_lookup_table.csv',index=False)
df_zoo=pd.merge(df_zoo.assign(EcoTaxa_hierarchy=lambda x: x.taxa.str.split(' ').str[0]),df_taxonomy,how='left',on=['EcoTaxa_hierarchy'])
# Volume-to-carbon conversion
df_zoo_volume=pd.read_csv(path_to_git / "data" / "datafiles" / "ola" / "extraction_zooplancton_1970_2025" / "leman_biovolumes_1738849437120-879640.csv",sep=';',decimal=',',encoding='latin-1',engine='python')
df_zoo_volume.columns=['project','site','station','datetime','sampling_gear','measurement_gear','depth_min','depth_max','technician','settled_volume']
df_zoo_volume['biomass_concentration']= ((df_zoo_volume['settled_volume']/0.0012)/df_zoo_volume.depth_max)/df_zoo_volume.depth_max# 1mg plankton biomass = 0.0012 ml
df_zoo_volume['sampling_datetime']=pd.to_datetime(df_zoo_volume.datetime,format='%d/%m/%Y')


plot=(ggplot(df_zoo_volume.dropna(subset='biomass_concentration'))+geom_line(mapping=aes(x='sampling_datetime',y='biomass_concentration'))+scale_y_continuous(trans='sqrt')+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.set_size_inches(11.4, 4.5)
plot.show()
plot.savefig(fname='{}/figures/ola/ts_all_{}.pdf'.format(str(path_to_git), variable), dpi=300, bbox_inches='tight')

df_test=df_zoo[df_zoo.Phylum.isna()]
df_summary=df_zoo.pivot_table(values=['concentration'],columns=['Order'],index=['sampling_date', 'depth_min', 'depth_max'],aggfunc=np.sum).reset_index()
df_summary['total_biomass_zoo']=df_summary.concentration.sum(axis=1)/df_summary.depth_max
df_summary=df_summary.reset_index()
df_summary.columns=[col[-1] if col[-1]!='' else col[0] for col in df_summary.columns]
df_summary['sampling_datetime']=pd.to_datetime(df_summary.sampling_date,format='%d/%m/%Y')
variable='total_biomass_zoo'
plot=(ggplot(df_summary)+geom_line(mapping=aes(x='sampling_datetime',y=variable))+scale_y_continuous(trans='sqrt')+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.set_size_inches(12, 4.5)
plot.show()
plot.savefig(fname='{}/figures/ola/ts_all_{}.pdf'.format(str(path_to_git), variable), dpi=300, bbox_inches='tight')

taxa=['Calanoida','Cyclopoida','Anomopoda','Ploima']
plot=(ggplot(df_summary[['sampling_datetime']+taxa].assign(month=lambda x: x.sampling_datetime.dt.month).melt(id_vars='month',value_vars=taxa))+
      facet_wrap('~variable',ncol=2)+
      geom_boxplot(mapping=aes(x='month',y='value',group='month'))+scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.set_size_inches(6, 6.5)
plot.show()
plot.savefig(fname='{}/figures/ola/climatology_{}.pdf'.format(str(path_to_git),'_'.join(taxa)), dpi=300, bbox_inches='tight')

