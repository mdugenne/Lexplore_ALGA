## Objective: This script performs a set of data processing steps on the initial test runs acquired with the CytoSense

## Workflow starts here:

# Load listmode datafiles
datafiles=natsorted(list(Path(path_to_git /'data' /'datafiles'/ 'cytosense').expanduser().rglob('*_Listmode.csv')))
df_data=pd.concat(map(lambda file: pd.read_csv(file,sep=r',',engine='python',encoding='latin-1').assign(File=file.parent.name),datafiles),axis=0)
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

