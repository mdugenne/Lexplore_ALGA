## Objective: This script performs a set of data processing steps on the initial test runs acquired with the CytoSense
import pandas as pd
from plotnine import geom_abline

from scripts.funcs_image_processing import cfg_metadata

## Workflow starts here:

# Testing the frame rate correction for volume imaged
## Load table including the different frame rates used to image the same sample
df_metadata=pd.read_excel(path_to_git / 'data' / 'Preliminary_test_framerate_volume_correction.xlsx')
## Load ecotaxa table (run script_cytosense.py) and merge to listmodes to convert scatter signals into actual size for particles not imaged
df_metadata['Path_ecotaxa']=df_metadata.File.apply(lambda file: Path(path_to_network /'lexplore' / 'LeXPLORE' / 'ecotaxa' / file / 'ecotaxa_table_{}.tsv'.format(file)))
df_ecotaxa=pd.concat(map(lambda file: pd.read_table(file,header=0,skiprows=range(1,2),encoding='latin-1',sep=r'\t').assign(File=file.parent.name) if file.exists() else pd.DataFrame({'File':file.parent.name},index=[0]),df_metadata.Path_ecotaxa.unique())).reset_index(drop=True)
df_ecotaxa['Particle ID']=df_ecotaxa.object_id.astype(str).str.rsplit('_',n=1).str[1]
df_metadata['Path_listmodes']=df_metadata.File.apply(lambda file: Path(path_to_network /'lexplore' / 'LeXPLORE' / 'export files' / 'IIF' / str('Export_'+file.rsplit('_',2)[0]+' '+' '.join(file.rsplit('_',2)[1:])) / '{}_Listmode.csv'.format(file.rsplit('_',2)[0]+' '+' '.join(file.rsplit('_',2)[1:]))))
df_listmode=pd.concat(map(lambda file: pd.read_csv(file,sep=r',',engine='python',encoding='utf-8').assign(File=file.parent.name.split('_',1)[1].replace(' ','_')) if file.exists() else pd.DataFrame({'File':file.parent.name.split('_',1)[1].replace(' ','_')},index=[0]),df_metadata.Path_listmodes.unique())).reset_index(drop=True)
df_ecotaxa=pd.merge(df_ecotaxa.astype({'Particle ID':str,'File':str}),df_listmode.astype({'Particle ID':str,'File':str}),how='left',on=['File','Particle ID'])

## Plot regression with statistics
df_fws=df_ecotaxa.assign(Area=lambda x: x.object_area*(cfg_metadata['pixel_size_cytosense']**2),ECD=lambda x: x.object_equivalent_diameter_area*(cfg_metadata['pixel_size_cytosense'])).query('ECD<{}'.format(cfg_metadata['height_cytosense']*(pixel_size**2)/4))[['File','Particle ID','Area','ECD']+[col for col in df_ecotaxa.columns if 'SWS' in col][0:3]]
df_fws=df_ecotaxa.assign(Area=lambda x: x.object_area*(cfg_metadata['pixel_size_cytosense']**2),ECD=lambda x: x.object_equivalent_diameter_area*(cfg_metadata['pixel_size_cytosense'])).query('ECD<{}'.format(cfg_metadata['height_cytosense']*(pixel_size**2)/4))[['File','Particle ID','Area','ECD']+[col for col in df_ecotaxa.columns if 'FWS' in col][0:3]]
x_var='SWS Total'
y_var,y_unit='ECD',r'$\mu$m'
slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df_fws.dropna()[x_var]),y=np.log10(df_fws.dropna()[y_var]))
plot = (ggplot(data=df_fws)+
        geom_abline(slope=slope, intercept=intercept, alpha=1) +
        geom_point(aes(x=x_var, y=y_var),size = 1, alpha=0.1, shape = 'o')+
        labs(x='{} (arbitrary units)'.format(x_var), y=r'{} ({})'.format(y_var,y_unit))+
        annotate('text', label='y = '+str(np.round(10**(-10**slope), 6))+'x '+str(np.round(10**(-10**intercept), 2))+ ', R$^{2}$ = '+str(np.round(r_value, 3)),  x = np.nanquantile(df_fws[x_var],[0.999]),y = np.nanquantile(df_fws[y_var],[0.999]),ha='right')+
        scale_x_log10(breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+07)), step=1), labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int( np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l])+
        scale_y_log10()+
        theme_paper).draw(show=False)
plot.savefig(fname='{}/figures/Initial_test/cytosense_size_calibration_{}.pdf'.format(str(path_to_git),x_var.replace(' ','_')), dpi=600)
x_var='SWS Length'
y_var,y_unit='ECD',r'$\mu$m'
slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df_fws.dropna()[x_var]),y=np.log10(df_fws.dropna()[y_var]))
plot = (ggplot(data=df_fws)+
        geom_abline(slope=slope, intercept=intercept, alpha=1) +
        geom_point(aes(x=x_var, y=y_var),size = 1, alpha=0.1, shape = 'o')+
        labs(x='{} (arbitrary units)'.format(x_var), y=r'{} ({})'.format(y_var,y_unit))+
        annotate('text', label='y = '+str(np.round(10**(-10**slope), 6))+'x '+str(np.round(10**(-10**intercept), 2))+ ', R$^{2}$ = '+str(np.round(r_value, 3)),  x = np.nanquantile(df_fws[x_var],[0.999]),y = np.nanquantile(df_fws[y_var],[0.999]),ha='right')+
        scale_x_log10(breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+07)), step=1), labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int( np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l])+
        scale_y_log10()+
        theme_paper).draw(show=False)
plot.savefig(fname='{}/figures/Initial_test/cytosense_size_calibration_{}.pdf'.format(str(path_to_git),x_var.replace(' ','_')), dpi=600)

## Compute the Normalized Biovolume Size Spectrum for all particles in the image-in-flow window
df_fws=df_ecotaxa.assign(Area=lambda x: x.object_area*(cfg_metadata['pixel_size_cytosense']**2),ECD=lambda x: x.object_equivalent_diameter_area*(cfg_metadata['pixel_size_cytosense'])).query('ECD<{}'.format(cfg_metadata['height_cytosense']*(pixel_size**2)/4))[['File','Particle ID','Area','ECD']+[col for col in df_ecotaxa.columns if 'FWS' in col][0:3]]
x_var='FWS Total'
y_var,y_unit='ECD',r'$\mu$m'
slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df_fws.dropna()[x_var]),y=np.log10(df_fws.dropna()[y_var]))
df_listmode['ECD'],df_listmode['area']=(df_listmode[x_var]**slope)*(10**intercept),(np.pi*(((df_listmode[x_var]**slope)*(10**intercept))/2)**2)/(cfg_metadata['pixel_size_cytosense']**2)
df_nbss_sample, df_nbss_boot_sample = nbss_estimates(df=pd.merge(df_listmode,df_ecotaxa[['File', 'sample_volume_analyzed_ml']].drop_duplicates().rename(columns={'sample_volume_analyzed_ml':'volume'}),how='left', on=['File']), pixel_size=cfg_metadata['pixel_size_cytosense'], grouping_factor=['File'])

plot = (ggplot(df_nbss_sample) +
        geom_ribbon(mapping=aes(x='size_class_mid', y='NBSS',ymin='np.maximum(0,NBSS-NBSS_std/2)',ymax='NBSS+NBSS_std/2',group='File',color='File'),alpha=0.1)+
        geom_point(mapping=aes(x='size_class_mid', y='NBSS',group='Group_index',colour='File'), alpha=1)+
        labs(x='Equivalent circular diameter ($\mu$m)',y='Normalized Biovolume Size Spectra ($\mu$m$^{3}$ mL$^{-1}$ $\mu$m$^{-3}$)', title='',colour='') +
        scale_y_log10(limits=[0.1,10000],breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+04)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        scale_x_log10(limits=[4,200],  breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e+00)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e+00)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(), labels=lambda l: [v if ((v / (10 ** np.floor(np.log10(v)))) == 1) else '' for v in l]) +
        guides(colour=None,fill=None)+
        theme_paper).draw(show=False)

plot.savefig(fname='{}/figures/Initial_test/cytosense_nbss_uncorrected_test.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')
## Add correction fluid imaged
df_listmode=pd.merge(df_listmode,df_ecotaxa[['File', 'sample_volume_analyzed_ml','sample_duration_sec','acq_flow_rate']].drop_duplicates(),how='left', on=['File'])
df_listmode=pd.merge(df_listmode,df_metadata[['File', 'Frame_rate']].drop_duplicates(),how='left', on=['File'])

df_listmode['Fluid_imaged_ml']=np.pi*((reg_calibration.predict(pd.DataFrame({'Flow_rate':df_listmode['acq_flow_rate'].astype(float)}))*1e-03/2))*(frame_width*pixel_size*1e-03)*df_listmode.Frame_rate*df_listmode.sample_duration_sec.astype(float)*1e-03
df_listmode=pd.merge(df_listmode.astype({'Particle ID':str,'File':str}),df_ecotaxa.astype({'Particle ID':str,'File':str})[['Particle ID','File']+[col for col in df_ecotaxa.columns if col not in df_listmode.columns]],how='left', on=['File','Particle ID'])
df_listmode=df_listmode.assign(label=lambda x: np.where(x.object_id.isna(),'Not imaged','Imaged'))
df_listmode=df_listmode.assign(volume=lambda x: np.where(x.object_id.isna(),x.sample_volume_analyzed_ml,x.Fluid_imaged_ml))

## Only use early samples
list_files=['lexplore_lakewater_surface_smart_2024-10-28_16h38','lexplore_lakewater_surface_smart_2024-10-28_16h20', 'lexplore_lakewater_surface_smart_2024-10-28_16h03']#['lexplore_lakewater_surface_smart_2024-10-28_10h19', 'lexplore_lakewater_surface_smart_2024-10-28_10h37', 'lexplore_lakewater_surface_smart_2024-10-28_10h55','lexplore_lakewater_surface_smart_2024-10-28_11h18', 'lexplore_lakewater_surface_smart_2024-10-28_11h37']

df_nbss_sample, df_nbss_boot_sample = nbss_estimates(df=df_listmode.query('label=="Imaged"'), pixel_size=cfg_metadata['pixel_size_cytosense'], grouping_factor=['File','label'])
df_nbss_sample, df_nbss_boot_sample = nbss_estimates(df=df_listmode.query('label=="Not imaged"'), pixel_size=cfg_metadata['pixel_size_cytosense'], grouping_factor=['File','label'])
df_nbss_sample, df_nbss_boot_sample = nbss_estimates(df=df_listmode, pixel_size=cfg_metadata['pixel_size_cytosense'], grouping_factor=['File','label'])
df_nbss_sample['Group_index']= pd.merge(df_nbss_sample.drop(columns=['Group_index']), df_nbss_sample.drop_duplicates(subset=['File','label'], ignore_index=True)[['File','label']].reset_index().rename( {'index': 'Group_index'}, axis='columns'), how='left', on=['File','label']).Group_index.astype(str).values


plot = (ggplot(df_nbss_sample.query('File.isin(["{}"])'.format('","'.join(list_files)))) +
        facet_wrap('~File',ncol=1)+
        geom_ribbon(mapping=aes(x='size_class_mid', y='NBSS',ymin='np.maximum(0,NBSS-NBSS_std/2)',ymax='NBSS+NBSS_std/2',group='Group_index',color='File'),alpha=0.1)+
        geom_point(mapping=aes(x='size_class_mid', y='NBSS',group='Group_index',color='File'), alpha=1)+
        labs(x='Equivalent circular diameter ($\mu$m)',y='Normalized Biovolume Size Spectra ($\mu$m$^{3}$ mL$^{-1}$ $\mu$m$^{-3}$)', title='',colour='') +
        scale_y_log10(limits=[0.1,10000],breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+04)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        scale_x_log10(limits=[4,200], breaks=np.multiply( 10 ** np.arange(np.floor(np.log10(1e+00)), np.ceil(np.log10(1e+04)), step=1).reshape( int((np.ceil(np.log10(1e+00)) - np.floor(np.log10(1e+04)))), 1), np.arange(1, 10, step=1).reshape(1, 9)).flatten(), labels=lambda l: [v if ((v / (10 ** np.floor(np.log10(v)))) == 1) else '' for v in l]) +
        guides(color=None,fill=None)+
        theme_paper).draw(show=False)
plot.set_size_inches(6.5,10)
plot.savefig(fname='{}/figures/Initial_test/cytosense_nbss_corrected_test.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')
plot = (ggplot(df_listmode.query('File.isin(["{}"])'.format('","'.join(list_files[0:1])))) +
        facet_wrap('~File')+
      #  stat_density_2d(aes(x='SWS Total', y='FL Red Total',fill=after_stat('density')), geom='raster', contour=False,alpha=0.1)+
        geom_point(mapping=aes(x='FWS Total', y='FL Red Total',colour='label'), alpha=1, size=0.00001) +  #
        scale_colour_manual(limits=['Not imaged','Imaged'],values={'Not imaged':'#{:02x}{:02x}{:02x}{:02x}'.format(204, 204 , 204,10),'Imaged':'#{:02x}{:02x}{:02x}'.format(87,219,178)})+
        coord_fixed()+#scale_fill_gradientn(trans="log10", colors=sns.color_palette(sns.color_palette("BuPu",15).as_hex())[::-1], guide=guide_colorbar(direction="horizontal"), na_value='#ffffff00') +
        labs(x='FWS Total (a.u.)',y='FL Red Total (a.u.)', title='',colour='') +
        scale_y_log10(limits=[1e-1, 1e+06],breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+06)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        scale_x_log10(limits=[1e-1, 1e+06], breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+06)), step=1), labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int( np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        theme_paper).draw(show=False)


plot.savefig(fname='{}/figures/Initial_test/cytosense_test_cytograms.png'.format(str(path_to_git)), dpi=300, bbox_inches='tight')


# Scatterplots based on listmodes
# Load listmode datafiles
datafiles=natsorted(list(Path(path_to_git /'data' /'datafiles'/ 'cytosense').expanduser().rglob('*_Listmode.csv')))
df_data=pd.concat(map(lambda file: pd.read_csv(file,sep=r',',engine='python',encoding='utf-8').assign(File=file.parent.name),datafiles),axis=0)
df_data.columns


plot = (ggplot(df_data) +
        facet_wrap('~File')+
      #  stat_density_2d(aes(x='SWS Total', y='FL Red Total',fill=after_stat('density')), geom='raster', contour=False,alpha=0.1)+
        geom_point(mapping=aes(x='SWS Total', y='FL Red Total'), alpha=1, size=0.00001) +  #
        coord_fixed()+#scale_fill_gradientn(trans="log10", colors=sns.color_palette(sns.color_palette("BuPu",15).as_hex())[::-1], guide=guide_colorbar(direction="horizontal"), na_value='#ffffff00') +
        labs(x='FWS Total (a.u.)',y='FL Red Total (a.u.)', title='',colour='') +
        scale_y_log10(limits=[1e-1, 1e+06],breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+06)), step=1),labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int(np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        scale_x_log10(limits=[1e-1, 1e+06], breaks=10 ** np.arange(np.floor(np.log10(1e-01)) - 1, np.ceil(np.log10(1e+06)), step=1), labels=lambda l: ['10$^{%s}$' % int(np.log10(v)) if (np.log10(v)) / int( np.log10(v)) == 1 else '10$^{0}$' if v == 1 else '' for v in l]) +
        theme_paper).draw(show=False)


save_as_pdf_pages([plot],'{}/figures/Initial_test/cytosense_test_cytograms.pdf'.format(str(path_to_git)))
plot.savefig(fname='{}/figures/Initial_test/cytosense_test_cytograms.png'.format(str(path_to_git)), dpi=300, bbox_inches='tight')

