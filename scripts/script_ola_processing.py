## Objective: This script generates summary data and figures from the database available at https://si-ola.inrae.fr/si_lacs/index.jsf#
# Load modules and functions required for image processing
import numpy as np

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
df_phyto['season']=df_phyto.datetime.map(dict(map(lambda sampling_date: (sampling_date,season(pd.to_datetime(sampling_date),'north')),df_phyto['datetime'].unique())))
df_phyto['Season']=pd.Categorical(df_phyto.season,["Winter","Spring","Summer","Fall"],ordered=True)
df_phyto['month']=df_phyto.datetime.dt.strftime('%m')
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
    df_taxonomy=pd.read_csv(path_datafiles[1].parent / 'taxonomy_lookup_table.csv')
df_phyto=pd.merge(df_phyto.assign(EcoTaxa_hierarchy=lambda x: x.taxa.str.split(' ').str[0].str.replace('cées','ceae')),df_taxonomy,how='left',on=['EcoTaxa_hierarchy'])
# Volume-to-carbon conversion
df_test=df_phyto[df_phyto.Class.isna()]
df_phyto['biomass_concentration']=np.where(df_phyto.Class=='Bacillariophyceae',1e-03*(10**-0.933)*df_phyto.biovolume_concentration**0.881,1e-03*(10**-0.665)*df_phyto.biovolume_concentration**0.939)
df_summary=df_phyto.groupby(['sampling_date','Class']).biomass_concentration.sum().reset_index().pivot_table(index=['sampling_date'],columns=['Class']).reset_index()
df_summary=df_phyto.pivot_table(values=['biomass_concentration'],columns=['Class'],index=['sampling_date', 'depth_min', 'depth_max'],aggfunc=np.sum).reset_index()
df_summary['total_biomass']=df_summary.biomass_concentration.sum(axis=1)
df_summary=df_summary.reset_index()
df_summary.columns=[col[-1] if col[-1]!='' else col[0] for col in df_summary.columns]
df_summary['sampling_datetime']=pd.to_datetime(df_summary.sampling_date,format='%d/%m/%Y')
df_summary=df_summary.sort_values('sampling_datetime').reset_index(drop=True)
taxa=['Cyanophyceae','Cryptophyceae','Bacillariophyceae','Chlorophyceae','Chrysophyceae','Zygnematophyceae']
palette_taxa=dict(zip(taxa,['#{:02x}{:02x}{:02x}'.format(111,145,138),'#{:02x}{:02x}{:02x}'.format(222,135,26),'#{:02x}{:02x}{:02x}'.format(170,136,0),'#{:02x}{:02x}{:02x}'.format(34,136,0),'#{:02x}{:02x}{:02x}'.format(128,102,0),'#{:02x}{:02x}{:02x}'.format(222,135,135)]))
variable='total_biomass'

#Class-specific climatologies
df_summary['periods']=pd.cut(df_summary.sampling_datetime,include_lowest=True,bins=[np.datetime64('1970-01-01'),np.datetime64('1986-01-01'),np.datetime64('2006-01-01'),np.datetime64('2026-01-01'),],labels=['Hypertrophic','Eutrophic','Mesotrophic']).astype(str)
plot=(ggplot(df_summary[['sampling_datetime']+taxa].assign(month=lambda x: x.sampling_datetime.dt.month).melt(id_vars='month',value_vars=taxa))+
      facet_wrap('~variable',ncol=2)+
      geom_boxplot(mapping=aes(x='month',y='value',group='month'))+scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.set_size_inches(6, 10.5)
plot.show()
plot.savefig(fname='{}/figures/ola/climatology_{}.pdf'.format(str(path_to_git),'_'.join(taxa)), dpi=300, bbox_inches='tight')

id=0
# Top-5 genus
df_summary=df_phyto.groupby(['sampling_date','Class','Genus']).biomass_concentration.sum().reset_index().groupby(['Class','Genus']).biomass_concentration.max().reset_index().groupby(['Class']).apply(lambda x: x.nlargest(10,'biomass_concentration'))
plot=(ggplot(df_summary.query('Class=="{}"'.format(taxa[id])).sort_values('biomass_concentration'))+
      theme(axis_text_x=element_text(rotation=45,hjust=1))+
      geom_bar(mapping=aes(x='factor(Genus,["{}"])'.format('","'.join(df_summary.query('Class=="{}"'.format(taxa[id])).Genus)),y='biomass_concentration'),stat='identity')+scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.show()
plot.savefig(fname='{}/figures/ola/top10_{}.pdf'.format(str(path_to_git),taxa[id]), dpi=300, bbox_inches='tight')

plot=(ggplot(df_summary[['sampling_datetime','periods']+taxa].assign(month=lambda x: x.sampling_datetime.dt.month).astype({'month':str}).melt(id_vars=['month','periods'],value_vars=taxa[id]))+
      facet_wrap('~variable',ncol=2)+
      scale_fill_manual(dict(zip(['Hypertrophic','Eutrophic','Mesotrophic'],list(map(lambda col: col.hex(),Color(palette_taxa[taxa[id]] ).gradient(to=Color('#ffffffff'), steps=3))))))+
      scale_x_continuous(breaks=np.arange(1,12,2))+
      geom_boxplot(mapping=aes(x='month.astype(float)',y='value',group="month + '.' + periods",fill='periods'),position=position_dodge(0))+scale_y_sqrt()+theme_paper + theme(legend_position='bottom',text=element_text(size=22))).draw(show=False)
plot.show()
plot.savefig(fname='{}/figures/ola/climatology_{}.pdf'.format(str(path_to_git),taxa[id]), dpi=300, bbox_inches='tight')

#Entire time-series
plot=(ggplot(df_summary)+geom_line(mapping=aes(x='sampling_datetime',y=variable))+scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot=(ggplot(df_summary)+geom_line(mappin
      geom_area(mapping=aes(x='sampling_datetime',y=taxa[0]),color='#{:02x}{:02x}{:02x}'.format(111,145,138),fill='#{:02x}{:02x}{:02x}'.format(111,145,138),alpha=0.5)+
    #  geom_area(mapping=aes(x='sampling_datetime',y=taxa[1]),color='#{:02x}{:02x}{:02x}'.format(222,135,26),fill='#{:02x}{:02x}{:02x}'.format(229,145,125),alpha=0.5)+
    #  geom_area(mapping=aes(x='sampling_datetime',y=taxa[2]),color='#{:02x}{:02x}{:02x}'.format(170,136,0),fill='#{:02x}{:02x}{:02x}'.format(170,136,0),alpha=0.5)+
    #  geom_area(mapping=aes(x='sampling_datetime',y=taxa[3]),color='#{:02x}{:02x}{:02x}'.format(34,136,0),fill='#{:02x}{:02x}{:02x}'.format(34,136,0),alpha=0.5)+
    #  geom_area(mapping=aes(x='sampling_datetime',y=taxa[4]),color='#{:02x}{:02x}{:02x}'.format(128,102,0),fill='#{:02x}{:02x}{:02x}'.format(128,102,0),alpha=0.5)+
    #  geom_area(mapping=aes(x='sampling_datetime',y=taxa[5]),color='#{:02x}{:02x}{:02x}'.format(222,135,135),fill='#{:02x}{:02x}{:02x}'.format(222,135,135),alpha=0.5)+
      scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot=(ggplot(df_summary.melt(id_vars=['sampling_datetime','total_biomass'],value_vars=taxa))+scale_fill_manual(palette_taxa)+
      geom_bar(mapping=aes(x='sampling_datetime',y='value',fill='variable'),stat='identity',width=50,alpha=0.7)+geom_line(data=df_summary,mapping=aes(x='sampling_datetime',y=variable))+
      scale_y_sqrt()+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.set_size_inches(12, 4.5)
plot.show()
plot.savefig(fname='{}/figures/ola/ts_all_{}.pdf'.format(str(path_to_git), variable), dpi=300, bbox_inches='tight')

df_zoo=pd.read_csv(path_datafiles[4],sep=';',decimal=',',encoding='latin-1',engine='python')
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
df_test=df_zoo[df_zoo.Class.isna()]
df_summary=df_zoo.pivot_table(values=['concentration'],columns=['Order'],index=['sampling_date', 'depth_min', 'depth_max'],aggfunc=np.sum).reset_index()
df_summary['total_biomass_zoo']=df_summary.concentration.sum(axis=1)
df_summary=df_summary.reset_index()
df_summary.columns=[col[-1] if col[-1]!='' else col[0] for col in df_summary.columns]
df_summary['sampling_datetime']=pd.to_datetime(df_summary.sampling_date)
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

