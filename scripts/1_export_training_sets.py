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
df_taxonomy_ecotaxa = df_taxonomy_ecotaxa[(df_taxonomy_ecotaxa.EcoTaxa_hierarchy != '') & (df_taxonomy_ecotaxa.EcoTaxa_hierarchy.isna() == False) & ( df_taxonomy_ecotaxa.EcoTaxa_hierarchy.astype(str) != 'nan')].reset_index(drop=True)

# Generate lookup taxonomy table
if not Path(path_to_git /'data' / 'taxonomy_lookup_table.csv').exists():
    df_taxonomy=pd.concat(map(lambda hierarchy: annotation_in_WORMS(hierarchy),df_taxonomy_ecotaxa.EcoTaxa_hierarchy)).reset_index(drop=True)
    while len(df_taxonomy)!=len(df_taxonomy_ecotaxa.EcoTaxa_hierarchy):
        ind=np.where(pd.Series(df_taxonomy_ecotaxa.EcoTaxa_hierarchy).isin(df_taxonomy.EcoTaxa_hierarchy.tolist())==False)
         for hierarchy in natsorted(df_taxonomy_ecotaxa.EcoTaxa_hierarchy[ind]):
            print(hierarchy)
            try:
                df_taxonomy=pd.concat([df_taxonomy,annotation_in_WORMS(hierarchy)],axis=0).reset_index(drop=True)
            except:
                pass
    df_taxonomy=df_taxonomy[(df_taxonomy.EcoTaxa_hierarchy!='') & (df_taxonomy.EcoTaxa_hierarchy.isna()==False) & (df_taxonomy.EcoTaxa_hierarchy.astype(str)!='nan')].reset_index(drop=True)
    df_taxonomy.astype({'Full_hierarchy':str}).query('Full_hierarchy!="nan"').to_csv(Path(path_to_git /'data' / 'taxonomy_lookup_table.csv'),index=False)
else:
    df_taxonomy=pd.read_csv(Path(path_to_git /'data' / 'taxonomy_lookup_table.csv'))
    df_taxonomy=df_taxonomy[(df_taxonomy.EcoTaxa_hierarchy!='') & (df_taxonomy.EcoTaxa_hierarchy.isna()==False) & (df_taxonomy.EcoTaxa_hierarchy.astype(str)!='nan')].reset_index(drop=True)
    ind = np.where(pd.Series(df_taxonomy_ecotaxa.EcoTaxa_hierarchy).isin(df_taxonomy.EcoTaxa_hierarchy.tolist()) == False)[0]
    while len(ind)>0:
        hierarchy=df_taxonomy_ecotaxa.EcoTaxa_hierarchy[ind].values[0]
        print(hierarchy)
        try:
            df_taxonomy = pd.concat([df_taxonomy, annotation_in_WORMS(hierarchy)], axis=0).reset_index(drop=True)
        except:
            df_taxonomy = pd.concat([df_taxonomy, pd.DataFrame({'EcoTaxa_hierarchy':hierarchy},index=[0])], axis=0).reset_index(drop=True)
        ind=ind[1:]
    df_taxonomy.astype({'Full_hierarchy': str}).query('Full_hierarchy!="nan"').sort_values(['EcoTaxa_hierarchy']).reset_index(drop=True).to_csv(Path(path_to_git / 'data' / 'taxonomy_lookup_table.csv'), index=False)