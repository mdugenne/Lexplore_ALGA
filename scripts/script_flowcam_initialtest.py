## Objective: This scripts performs a set of data processing steps on the initial test runs acquired with the FlowCam

import warnings
warnings.filterwarnings(action='ignore')


# Load modules and functions required for image processing

try:
    from funcs_image_processing import *

except:
    from scripts.funcs_image_processing import *

#Workflow starts here
path_to_network=Path("R:") # Set working directory to forel-meco
# Load metadata files (volume imaged, background pixel intensity, etc.) and save entries into separate tables (metadata and statistics)
metadatafiles=list(Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data').expanduser().rglob('Flowcam_10x_lexplore*/*_summary.csv'))
metadatafiles=list(filter(lambda element: element.parent.parent.stem not in ['Tests','Blanks'],metadatafiles))
df_metadata=pd.concat(map(lambda file:(df:=pd.read_csv(file,sep=r'\t',engine='python',encoding='latin-1',quotechar=r'\"',names=['Name','Value']),df:=df.Name.str.split(r'\,',n=1,expand=True).rename(columns={0:'Name',1:'Value'}),df:=df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).dropna().reset_index(drop=True) ,(df:=df.assign(Name=lambda x: x.Name.str.replace('========','').str.replace(' ','_').str.strip('_'),Value=lambda x: x.Value.str.lstrip(' ')).set_index('Name').T.rename(index={'Value':file.parent.name})),df:=df[[col for col in df.columns if col in summary_metadata_columns]])[-1],metadatafiles),axis=0)
df_metadata['len_id_to_skip']=df_metadata.Skip.astype(str).apply(lambda ids: len(sum(list(map(lambda id: list(np.arange(int(re.sub('\W+', '', id.rsplit(':')[0])), int(re.sub('\W+', '', id.rsplit(':')[1])) + 1)) if len( id.rsplit(':')) == 2 else [int(re.sub('\W+', '', id))],ids.rsplit(r"]+["))), [])) if not ids=='nan' else 0)

#Check the fluid volume imaged calculation based on the number of frames used. If
df_metadata.Used.astype(float)/df_metadata.Fluid_Volume_Imaged.str.split(' ').str[0].astype(float)
# This should be constant if the area of acceptable region, the size calibration, and the depth of the flow cell are kept constant
df_particle_statistics=pd.concat(map(lambda file:(df:=pd.read_csv(file,sep=r'\t',engine='python',encoding='latin-1',names=['Name','Value']),df:=df.Name.str.split(r'\,',n=1,expand=True).rename(columns={0:'Name',1:'Value'}),df:=df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).reset_index(drop=True),df:=df[df.query('Name.str.contains(r"\=",case=True)').index[0]:].reset_index(drop=True) ,df_summary_statistics:=pd.DataFrame([(df.query('Name.str.contains(r"\=",case=True)').index).tolist(),((df.query('Name.str.contains(r"\=",case=True)').index)[1:]-1).tolist()+[len(df)]]).T.apply(lambda id:{df.loc[id[0],'Name'].split(' ')[1]:pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),df.loc[id[0]+1:id[1],'Value'].values.tolist()[1].split(','))),index=[file.parent.name]) if len(df.loc[id[0]+1:id[1],'Value'].values.tolist())>1 else pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),[np.nan]*len(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(',')))),index=[file.parent.name])},axis=1,result_type='reduce'),final_df:=list(df_summary_statistics[[id for id,key in df_summary_statistics.items() if 'Particle' in key.keys()][0]].values())[0] if len([id for id,key in df_summary_statistics.items() if 'Particle' in key.keys()]) else pd.DataFrame({},index=[file.parent.name]))[-1],metadatafiles),axis=0)
df_filter_statistics=pd.concat(map(lambda file:(df:=pd.read_csv(file,sep=r'\t',engine='python',encoding='latin-1',names=['Name','Value']),df:=df.Name.str.split(r'\,',n=1,expand=True).rename(columns={0:'Name',1:'Value'}),df:=df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).reset_index(drop=True),df:=df[df.query('Name.str.contains(r"\=",case=True)').index[0]:].reset_index(drop=True) ,df_summary_statistics:=pd.DataFrame([(df.query('Name.str.contains(r"\=",case=True)').index).tolist(),((df.query('Name.str.contains(r"\=",case=True)').index)[1:]-1).tolist()+[len(df)]]).T.apply(lambda id:{df.loc[id[0],'Name'].split(' ')[1]:pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),df.loc[id[0]+1:id[1],'Value'].values.tolist()[1].split(','))),index=[file.parent.name]) if len(df.loc[id[0]+1:id[1],'Value'].values.tolist())>1 else pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),[np.nan]*len(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(',')))),index=[file.parent.name])},axis=1,result_type='reduce'),final_df:=list(df_summary_statistics[[id for id,key in df_summary_statistics.items() if 'Filter' in key.keys()][0]].values())[0] if len([id for id,key in df_summary_statistics.items() if 'Filter' in key.keys()]) else pd.DataFrame({},index=[file.parent.name]))[-1],metadatafiles),axis=0)
df_summary_statistics=pd.concat(map(lambda file:(df:=pd.read_csv(file,sep=r'\t',engine='python',encoding='latin-1',names=['Name','Value']),df:=df.Name.str.split(r'\,',n=1,expand=True).rename(columns={0:'Name',1:'Value'}),df:=df.query('not Name.str.contains(r"\:|End",case=True)').drop(index=[0]).reset_index(drop=True),df:=df[df.query('Name.str.contains(r"\=",case=True)').index[0]:].reset_index(drop=True) ,df_summary_statistics:=pd.DataFrame([(df.query('Name.str.contains(r"\=",case=True)').index).tolist(),((df.query('Name.str.contains(r"\=",case=True)').index)[1:]-1).tolist()+[len(df)]]).T.apply(lambda id:{df.loc[id[0],'Name'].split(' ')[1]:pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),df.loc[id[0]+1:id[1],'Value'].values.tolist()[1].split(','))),index=[file.parent.name]) if len(df.loc[id[0]+1:id[1],'Value'].values.tolist())>1 else pd.DataFrame(dict(zip(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(','),[np.nan]*len(df.loc[id[0]+1:id[1],'Value'].values.tolist()[0].split(',')))),index=[file.parent.name])},axis=1,result_type='reduce'),final_df:=list(df_summary_statistics[[id for id,key in df_summary_statistics.items() if 'Metadata' in key.keys()][0]].values())[0] if len([id for id,key in df_summary_statistics.items() if 'Metadata' in key.keys()]) else pd.DataFrame({},index=[file.parent.name]))[-1],metadatafiles),axis=0)

# Filter first runs
df_summary_statistics.index
df_summary_statistics=df_summary_statistics.drop(index=['Flowcam_10x_lexplore_wasam_20240920_2024-09-27 03-50-05', 'Flowcam_10x_lexplore_wasam_20241002_2024-10-03 07-23-04'])
pattern_to_keep='Flowcam_10x_lexplore_wasam_20241102'
df_summary_statistics=df_summary_statistics.drop(index=[sample for sample in df_summary_statistics.index if pattern_to_keep not in sample])

# Plot abundance
df_summary_statistics['Analysis_time']=pd.to_datetime(np.where(df_summary_statistics['Start Time'].str.len()>0,pd.to_datetime(df_summary_statistics['Start Time'].str[0:19],format='%Y-%m-%dT%H:%M:%S'),pd.NaT))
df_summary_statistics['Analysis_time']=df_summary_statistics.Analysis_time.dt.floor('1d') # Round acquisition time to days
df_summary_statistics['Treatment']=pd.Categorical(pd.Series(list(df_summary_statistics.index.astype(str))).str[47:].str.replace('_replicate1','').str.replace('_replicate2','').str.replace('_replicate3','').str.replace('nofixative','No fixative').str.replace('glut','0.25% glutaraldehyde').str.replace('pluronic_glut','pluronic + glutaraldehyde'))
df_summary_statistics['Timepoint']=pd.Categorical('T'+((df_summary_statistics['Analysis_time']-df_summary_statistics.Analysis_time.min()).dt.days.astype(int).astype(str)),categories='T'+((df_summary_statistics['Analysis_time']-df_summary_statistics.Analysis_time.min()).dt.days.astype(int).astype(str)).unique(),ordered=True)#pd.Categorical('T'+(df_summary_statistics['Analysis_time']-df_summary_statistics.loc['Flowcam_10x_lexplore_wasam_20241002_2024-10-04_nofixative','Analysis_time']).dt.days.astype(int).astype(str),categories=['T0','T2','T4','T6','T9','T11'],ordered=True)
# Correct the abundance in summary statistics in case any ID to skip were found
df_summary_statistics=pd.merge(df_summary_statistics,df_metadata[['Fluid_Volume_Imaged','Skip','len_id_to_skip']],how='left',right_index=True,left_index=True)
df_summary_statistics['Abundance_uncorrected']=df_summary_statistics.Count.astype(float)/df_summary_statistics.Fluid_Volume_Imaged.astype(str).str[0:6].astype(float)#df_summary_statistics['Particles / ml'].astype(float)
df_summary_statistics['Abundance']=(df_summary_statistics.Count.astype(float)-df_summary_statistics.len_id_to_skip)/df_summary_statistics.Fluid_Volume_Imaged.astype(str).str[0:6].astype(float)

plot = (ggplot(df_summary_statistics) +
        stat_summary(mapping=aes(x='Timepoint', y='Abundance', group='Treatment',fill='Treatment'),color='black', alpha=1,geom='bar', position = 'dodge') + #
        stat_summary(mapping=aes(x='Timepoint', y='Abundance', group='Treatment',fill='Treatment'), width=0,position=position_dodge(.9), alpha=1,geom='errorbar') +
        labs(x='Timepoint',y='Particles mL$^{-1}$', title='',colour='') +
         scale_y_continuous() +scale_fill_manual(values={'No fixative':'grey','0.25% glutaraldehyde':'black','fresh':'#ffffff00','pluronic':'black','pluronic + glutaraldehyde':'grey'})+
        theme_paper).draw(show=False)
plot.savefig(fname='{}/figures/Initial_test/flowcam_test_fixative_withpluronic.svg'.format(str(path_to_git)), dpi=300, bbox_inches='tight')

df_summary_statistics['Abundance_min']=poisson.interval(0.95,mu=df_summary_statistics['Abundance'].astype(int))[0]
df_summary_statistics['Abundance_max']=poisson.interval(0.95,mu=df_summary_statistics['Abundance'].astype(int))[1]

plot = (ggplot(df_summary_statistics) +
        geom_bar(mapping=aes(x='Timepoint', y='Abundance', group='Treatment',fill='Treatment'),color='black', alpha=1,stat='identity', position = 'dodge') + #
        geom_errorbar(mapping=aes(x='Timepoint', y='Abundance', ymin='Abundance_min', ymax='Abundance_max', group='Treatment',fill='Treatment'), width=0,position=position_dodge(.9), alpha=1,stat='identity') +
        labs(x='Timepoint',y='Particles mL$^{-1}$', title='',colour='') +
         scale_y_continuous() +scale_fill_manual(values={'No fixative':'grey','0.25% glutaraldehyde':'black','fresh':'#ffffff00','pluronic':'black','pluronic + glutaraldehyde':'grey'})+
        theme_paper).draw(show=False)

plot.savefig(fname='{}/figures/Initial_test/flowcam_test_fixative_withpluronic.svg'.format(str(path_to_git)), dpi=300, bbox_inches='tight')

# Plot color intensity
datafiles=list(Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data').expanduser().rglob('Flowcam_10x_lexplore*/*.csv'))
datafiles=list(set(datafiles) - set(metadatafiles))
df_data=pd.concat(map(lambda file: pd.read_csv(file,sep=r',',engine='python',encoding='latin-1').assign(File=file.parent.name),datafiles),axis=0)
df_data['Pixel']=rgb2gray(df_data[['Average Red','Average Green','Average Blue']])
df_summary_data=pd.merge(df_summary_statistics,df_data.groupby(['File'])['Average Green'].agg('describe')[['25%', '50%', '75%', 'count']],how='left',left_index=True,right_on='File')

plot = (ggplot(df_summary_data) +
        geom_bar(mapping=aes(x='Timepoint', y='50%', group='Treatment',fill='Treatment'), alpha=1,stat='identity', position = 'dodge') + #
        geom_errorbar(mapping=aes(x='Timepoint', y='50%', ymin='25%', ymax='75%', group='Treatment',fill='Treatment'), width=0,position=position_dodge(.9), alpha=1,stat='identity') +
        labs(x='Timepoint',y='Mean pixel intensity (8-bit gray)', title='',colour='') +
         coord_cartesian(ylim=[160,180]) +scale_fill_manual(values={'No fixative':'grey','0.25% glutaraldehyde':'black'})+
        theme_paper).draw(show=False)

plot.savefig(fname='{}/figures/Initial_test/flowcam_test_fixative_color.svg'.format(str(path_to_git)), dpi=300, bbox_inches='tight')
