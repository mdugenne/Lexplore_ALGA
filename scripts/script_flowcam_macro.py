# conda activate /Users/dugennem/opt/anaconda3/envs/xesmf_env
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as animation
from functools import partial
import pandas as pd
from pathlib import Path
import os
from natsort import natsorted
import skimage as ski  # pip install -U scikit-image
from skimage import color, measure, morphology
from PIL import Image, ImageColor
from skimage.util import compare_images, crop
from skimage.color import rgb2gray
from skimage.measure import regionprops
import cv2
import torch
from tqdm import tqdm
import numpy as np
import scipy as sp

#Workflow starts here

path_to_dir=Path('~/GIT/Lexplore_ALGA/data/datafiles/flowcam/').expanduser()
os.chdir(path_to_dir)
sys.path.append(str(path_to_dir))
from sinking.funcs_image_utils import *

path_to_cnn=Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data' / 'Lexplore' / 'cnn' / 'input').expanduser()

path_to_runs=natsorted(list(path_to_cnn.glob('Flowcam_2mm_lexplore_wasam_2025*')))
# Select weekly run
mondays=pd.date_range(start='2025-01', end='2026-01', freq='W-MON').strftime('%Y%m%d').tolist()
path_to_runs=[run for run in path_to_runs if run.stem.split('_')[4] in mondays]
with tqdm(desc='Generating features for run {}'.format(''), total=len(path_to_runs)-1), bar_format='{desc}{bar}', position=0, leave=True) as pbar:
    for run in path_to_runs:
        # Extract features from CNN
        save_features_directory=run
        pbar.set_description('Generating features for run {}'.format(run.stem), refresh=True)
            if not Path(str(save_features_directory /'images_features.csv')).exists():
                df_features = image_feature_dataset(image_path=run,filter=[], model_name='efficientnet_b0', layer_name='avgpool')
                df_features
                df_features.to_csv(str(save_features_directory /'images_features.csv'),index=False)
            else:
                df_features = pd.read_csv(str(save_features_directory / 'images_features.csv' ) )

        pbar.update(1)

df_features=pd.concat(list(map(lambda file: pd.read_csv(file),natsorted(list(Path(path_to_runs[0].parent).glob('Flowcam_2mm_*/images_features.csv')))))).reset_index(drop=True)

# Project and Cluster image features
save_features_directory=path_to_runs[0].parent
df_features.to_csv(str(save_features_directory / 'Flowcam_2mm_images_features.csv'),index=False)
if not Path(str(save_features_directory /'Flowcam_2mm_images_features_projected.csv')).exists():
    model_cluster = TSNE(n_components=3)
    df_projection = pd.DataFrame(model_cluster.fit_transform(pd.DataFrame(normalize(df_features[df_features.columns[np.arange(2, df_features.shape[1])]]))))
    df_projection['image_url'] = df_features.path
    df_projection.rename(columns={'image_url': 'images'}).to_csv(str(save_features_directory / 'Flowcam_2mm_images_features_projected.csv' ) , index=False)
else:
    df_projection=pd.read_csv(str(save_features_directory / 'Flowcam_2mm_images_features_projected.csv' ))

# Check projected images with clusters
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)
import plotly.io as pio
pio.renderers.default = "browser"
app,df_clusters=scatter_2d_images(df_directory=str(save_features_directory / 'Flowcam_2mm_images_features_projected.csv' ))
app.run(port=8888, use_reloader=False,debug=True)
print('App running on local host: http://127.0.0.1:8888/.\nPress ctrl + c to exit')

# Plot projection
plotnine.options.base_margin=0.05
plot=(ggplot(df_clusters.melt(id_vars=['images','score','cluster','x'],value_vars=['y','z'],var_name='axis',value_name='value'))+facet_wrap('~axis')+labs(y='',x='',colour='Particle cluster:')+geom_point(mapping=aes(x='x',y='value',colour='factor(cluster)'),alpha=.2,size=5)+scale_color_manual(values=dict(zip(df_clusters.cluster.unique(),px.colors.qualitative.Plotly*int(np.ceil(df_clusters.cluster.nunique()/ 10)))))+theme_plot+
      theme(legend_position=(.5, -1.05),legend_direction='horizontal',plot_margin=1.7)+guides(color=guide_legend(direction='horizontal',nrow=3))).draw(show=True).set_size_inches(12,5)
plot.savefig(fname='{}/run_{}_projection.pdf'.format(str(save_features_directory),run.stem), dpi=300, bbox_inches='tight')

