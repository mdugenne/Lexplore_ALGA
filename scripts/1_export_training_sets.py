## Objective: This script export Ecotaxa image training sets and train a ML model for taxa identification

# Load modules and functions required for image processing
try:
    from funcs_image_processing import *
except:
    from scripts.funcs_image_processing import *

#Workflow starts here
list_of_projects={project.replace('ecotaxa_lexplore_alga_','').replace('_projectid',''):cfg_metadata[project] for project in cfg_metadata.keys() if '_projectid' in project}
dict_of_storage_directories={'cytosense':Path(path_to_network /'lexplore' / 'LeXPLORE' / 'ecotaxa' ),'flowcam_macro':Path(cfg_metadata['flowcam_macro_context_file'].replace('acquisitions','ecotaxa')).parent,'flowcam_micro':Path(cfg_metadata['flowcam_10x_context_file'].replace('acquisitions','ecotaxa')).parent}
# Export project using EcoTaxa API and standardize annotations using World Register of (Marine) Species:
for instrument,project in list_of_projects.items():
    export_ecotaxa_project_table(ecotaxa_configuration=configuration,project_id=project,export_path=dict_of_storage_directories[instrument])

df_taxonomy_ecotaxa=check_ecotaxa_annotations(ecotaxa_configuration=configuration,project_id=','.join(list(map(str,list_of_projects.values()))))
df_taxonomy_ecotaxa = df_taxonomy_ecotaxa[(df_taxonomy_ecotaxa.Ecotaxa_annotation_category != '') & (df_taxonomy_ecotaxa.Ecotaxa_annotation_category.isna() == False) & ( df_taxonomy_ecotaxa.Ecotaxa_annotation_category.astype(str) != 'nan')].reset_index(drop=True)

# Generate lookup taxonomy table
if not Path(path_to_git /'data' / 'taxonomy_lookup_table.csv').exists():
    df_taxonomy=pd.concat(map(lambda hierarchy: annotation_in_WORMS(hierarchy),df_taxonomy_ecotaxa.Ecotaxa_annotation_hierarchy)).reset_index(drop=True)
    while len(df_taxonomy)!=len(df_taxonomy_ecotaxa.Ecotaxa_annotation_hierarchy):
        ind=np.where(pd.Series(df_taxonomy_ecotaxa.Ecotaxa_annotation_hierarchy).isin(df_taxonomy.Ecotaxa_annotation_hierarchy.tolist())==False)
         for hierarchy in natsorted(df_taxonomy_ecotaxa.Ecotaxa_annotation_hierarchy[ind]):
            print(hierarchy)
            try:
                df_taxonomy=pd.concat([df_taxonomy,annotation_in_WORMS(hierarchy)],axis=0).reset_index(drop=True)
            except:
                pass
    df_taxonomy=df_taxonomy[(df_taxonomy.Ecotaxa_annotation_category!='') & (df_taxonomy.Ecotaxa_annotation_category.isna()==False) & (df_taxonomy.Ecotaxa_annotation_category.astype(str)!='nan')].reset_index(drop=True)
    df_taxonomy['Ecotaxa_annotation_category']=np.where(df_taxonomy.Ecotaxa_annotation_hierarchy.str.split('>').str[-1].isna(),df_taxonomy.Ecotaxa_annotation_category.str.split('>').str[-1],df_taxonomy.Ecotaxa_annotation_hierarchy.str.split('>').str[-1])
    df_taxonomy.astype({'Full_hierarchy':str}).query('Full_hierarchy!="nan"').sort_values('Ecotaxa_annotation_hierarchy').reset_index(drop=True).to_csv(Path(path_to_git /'data' / 'taxonomy_lookup_table.csv'),index=False)
else:
    df_taxonomy=pd.read_csv(Path(path_to_git /'data' / 'taxonomy_lookup_table.csv'))
    df_taxonomy=df_taxonomy[(df_taxonomy.Ecotaxa_annotation_category!='') & (df_taxonomy.Ecotaxa_annotation_category.isna()==False) & (df_taxonomy.Ecotaxa_annotation_category.astype(str)!='nan')].reset_index(drop=True)
    ind = np.where(pd.Series(df_taxonomy_ecotaxa.Ecotaxa_annotation_category).isin(df_taxonomy.Ecotaxa_annotation_category.tolist()) == False)[0]
    while len(ind)>0:
        hierarchy=df_taxonomy_ecotaxa.Ecotaxa_annotation_category[ind].values[0]
        print(hierarchy)
        try:
            df_taxonomy = pd.concat([df_taxonomy, annotation_in_WORMS(hierarchy)], axis=0).reset_index(drop=True)
        except:
            df_taxonomy = pd.concat([df_taxonomy, pd.DataFrame({'EcoTaxa_hierarchy':hierarchy},index=[0])], axis=0).reset_index(drop=True)
        ind=ind[1:]
    df_taxonomy['Ecotaxa_annotation_category']=np.where(df_taxonomy.Ecotaxa_annotation_hierarchy.str.split('>').str[-1].isna(),df_taxonomy.Ecotaxa_annotation_category.str.split('>').str[-1],df_taxonomy.Ecotaxa_annotation_hierarchy.str.split('>').str[-1])
    df_taxonomy.astype({'Full_hierarchy':str}).query('Full_hierarchy!="nan"').sort_values('Ecotaxa_annotation_hierarchy').reset_index(drop=True).to_csv(Path(path_to_git /'data' / 'taxonomy_lookup_table.csv'),index=False)
    df_taxonomy.astype({'Full_hierarchy': str}).query('Full_hierarchy!="nan"').sort_values(['EcoTaxa_hierarchy']).reset_index(drop=True).to_csv(Path(path_to_git / 'data' / 'taxonomy_lookup_table.csv'), index=False)

# Load latest ecotaxa table
path_ecotaxa_files={instrument: natsorted(list(dict_of_storage_directories[instrument].glob('ecotaxa_export_*_{}_*.tsv'.format(list_of_projects[instrument]))))[-1] if len(natsorted(list(dict_of_storage_directories[instrument].glob('ecotaxa_export_*_{}_*.tsv'.format(list_of_projects[instrument]))))) else None for instrument in dict_of_storage_directories.keys()}
df_ecotaxa=pd.concat(map(lambda file:pd.read_csv(file,sep='\s+',encoding='latin-1') if not file is None else pd.DataFrame(),list(path_ecotaxa_files.values()))).reset_index(drop=True)
df_ecotaxa=df_ecotaxa.rename(columns={df_ecotaxa.columns[0]:'object_id'})
df_ecotaxa=pd.merge(df_ecotaxa,df_taxonomy,how='left',left_on='object_annotation_category',right_on='Ecotaxa_annotation_category')
df_ecotaxa.query('object_annotation_category=="Planktothrix"')[['object_equivalent_diameter_area','object_axis_minor_length','object_axis_major_length']].agg(['mean','std']).T

# Relative abundance of main phytoplankton classes
natsorted(df_ecotaxa.Class.unique())

df_summary=df_ecotaxa.assign(Class_grouped=np.where(df_ecotaxa.Class.isin(['Trebouxiophyceae','Xanthophyceae','Zygnematophyceae']),'Zygnematophyceae',df_ecotaxa.Class)).query('Class_grouped.isin(["Bacillariophyceae","Chlorophyceae","Chrysophyceae","Zygnematophyceae","Cyanophyceae","Dinophyceae"])').groupby(['object_date','Class_grouped']).apply(lambda x: pd.Series({'Count':x.object_id.count(),'Volume_imaged':x.sample_volume_fluid_imaged_ml.unique()[0],'Abundance':x.object_id.count()/x.sample_volume_fluid_imaged_ml.unique()[0],'Biovolume_concentration':sum((1/6)*np.pi*x.object_equivalent_diameter_area**3)/x.sample_volume_fluid_imaged_ml.unique()[0]})).reset_index()
df_summary['datetime']=pd.to_datetime(df_summary.object_date.astype(str).str.zfill(6), format='%Y%m%d')
df_summary['week']= df_summary.datetime.dt.strftime('%Y %B %w')
df_summary['relative_abundance']=df_summary.Abundance/df_summary.groupby(['object_date']).Abundance.transform('sum')
df_summary['relative_biovolume_concentration']=df_summary.Biovolume_concentration/df_summary.groupby(['object_date']).Biovolume_concentration.transform('sum')
df_summary['weekly_datetime']=df_summary.datetime.dt.to_period('W').dt.end_time
taxa=['Cyanophyceae','Cryptophyceae','Bacillariophyceae','Chlorophyceae','Chrysophyceae','Zygnematophyceae']
palette_taxa=dict(zip(taxa,['#{:02x}{:02x}{:02x}'.format(111,145,138),'#{:02x}{:02x}{:02x}'.format(222,135,26),'#{:02x}{:02x}{:02x}'.format(170,136,0),'#{:02x}{:02x}{:02x}'.format(34,136,0),'#{:02x}{:02x}{:02x}'.format(128,102,0),'#{:02x}{:02x}{:02x}'.format(85,0,0)]))
id=0
plot = (ggplot(df_summary.query('Class_grouped=="{}"'.format(taxa[id]))) + #.groupby(['weekly_datetime','Class_grouped']).relative_biovolume_concentration.agg('describe')[['25%','50%','75%']].reset_index()
        #geom_pointrange(mapping=aes(x='weekly_datetime',y='50%',ymin='25%',ymax='75%',fill='Class_grouped'),size=.7,alpha=0.8)+
        geom_violin(mapping=aes(x='weekly_datetime',y='relative_biovolume_concentration',fill='Class_grouped',group='weekly_datetime'),trim=True,bw=.01)+
        #geom_line(mapping=aes(x='weekly_datetime',y='50%',ymin='25%',ymax='75%',colour='Class_grouped'),size=.9,alpha=0.2,linetype='dotted')+
        scale_fill_manual(values=palette_taxa)+scale_colour_manual(values=palette_taxa)+
        labs(x='',y='Relative abundance') +
        scale_x_datetime(limits=pd.to_datetime(['2024-12-01','2025-09-01']))+
        scale_y_sqrt(limits=[0,1])+
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)
plot.set_size_inches(10.7, 4.5)
#plot.show()
plot.savefig(fname='{}/figures/ecotaxa/succession_{}.pdf'.format(str(path_to_git),taxa[id]), dpi=300, bbox_inches='tight')


# Save summary table with readme file
((df_summary[df_summary.datetime>pd.to_datetime('2025-01-01')].rename(columns={'Class_grouped':'Class'}))[['datetime','Class','Volume_imaged','Count','Abundance','Biovolume_concentration','relative_abundance','relative_biovolume_concentration']]).to_csv(Path(path_to_git / 'data' / 'ecotaxa_export_summary_{}_{}.csv'.format(path_ecotaxa_files['flowcam_micro'].stem.split('_TSV_')[1].split('_')[0],path_ecotaxa_files['flowcam_micro'].stem.split('_TSV_')[1].split('_')[1])),index=False)
pd.DataFrame({'column':['datetime','Class','Volume_imaged','Count','Abundance','Biovolume_concentration','relative_abundance','relative_biovolume_concentration'],
              'unit/format':['yyyy-mm-dd','','milliliters','#','count per milliliters','cubic micrometers per milliliters','unitless [0-1]','unitless [0-1]'],
              'description':['Sampling date. Sample collection occurs daily at 09:00','Taxonomic class according to the World Register of Marine Species','Volume imaged during Flowcam acquisitions. Should be cumulated for weekly statistics','single-cell particles or colonies','concentration of single-cell or colonial organisms','Biovolume concentration that accounts for difference between single-cell and colonial organism sizes','Relative abundance of the different taxonomic classes in each sample','Relative biovolume concentration of the different taxonomic classes in each sample']}).to_csv(Path(path_to_git / 'data' / 'readme_ecotaxa_export_summary_{}_{}.csv'.format(path_ecotaxa_files['flowcam_micro'].stem.split('_TSV_')[1].split('_')[0],path_ecotaxa_files['flowcam_micro'].stem.split('_TSV_')[1].split('_')[1])),index=False)

# Check diatom successions
df_summary=df_ecotaxa.query('Class=="Bacillariophyceae"').groupby(['object_date','Class','Genus','Ecotaxa_annotation_category']).apply(lambda x: pd.Series({'Count':x.object_id.count(),'Abundance':x.object_id.count()/x.sample_volume_fluid_imaged_ml.unique()[0]})).reset_index()
df_summary['datetime']=pd.to_datetime(df_summary.object_date.astype(str).str.zfill(6), format='%Y%m%d')

plot = (ggplot(df_summary[df_summary.Ecotaxa_annotation_category.str.contains('chytrid')]) +
        #geom_point(data=df_summary.query('Genus.isin(["Asterionella"])'),mapping=aes(x='datetime',y='Abundance',group='EcoTaxa_hierarchy',colour='EcoTaxa_hierarchy'))+
        geom_point(mapping=aes(x='datetime',y='Abundance',colour='Ecotaxa_annotation_category'))+
        geom_smooth(mapping=aes(x='datetime',y='Abundance',colour='Ecotaxa_annotation_category'),span=.01)+
        labs(x='',y='Abundance (infected colony mL$^{-1}$)') +
        scale_x_datetime(limits=pd.to_datetime(['2025-01-01','2025-06-01']))+
        scale_y_log10(breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e-03)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e-03)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)
plot.set_size_inches(6.7, 3.5)
plot.show()
plot.savefig(fname='{}/figures/ecotaxa/succession_infection.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')



df_stat=df_summary.query('Genus.isin(["Diatoma"])').assign(week=lambda x: x.datetime.dt.strftime('%Y %B %w')).groupby(['week','Ecotaxa_annotation_category']).apply(lambda x: pd.Series({'datetime':x.datetime.median(),'Abundance_min':x.Abundance.quantile(0.05),'Abundance':x.Abundance.quantile(0.5),'Abundance_max':x.Abundance.quantile(0.95)})).reset_index()

plot = (ggplot() +
        #geom_point(data=df_summary.query('Genus.isin(["Asterionella"])'),mapping=aes(x='datetime',y='Abundance',group='EcoTaxa_hierarchy',colour='EcoTaxa_hierarchy'))+
        geom_pointrange(data=df_stat,mapping=aes(x='datetime',y='Abundance',ymin='Abundance_min',ymax='Abundance_max',colour='Ecotaxa_annotation_category'))+
        scale_colour_gray()+
        labs(x='',y='Abundance (colony mL$^{-1}$)') +
        scale_x_datetime(limits=pd.to_datetime(['2025-01-01','2025-06-01']))+
        scale_y_log10(breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e-03)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e-03)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)
plot.show()