import cv2
import torch #pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
import imagesize
import numpy  as np
import pandas as pd
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import sys
from tqdm import tqdm
## interactive module
import dash
from dash.exceptions import PreventUpdate
from dash import Dash,dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import base64
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from itertools import compress
from sklearn.neighbors import NearestNeighbors

# Path and File processing modules
import os
import scipy as sp
from pathlib import Path
path_to_git=Path('~/GIT/Lexplore_ALGA').expanduser()
# Read the metadata stored in yaml file
import yaml #conda install PyYAML
path_to_config = path_to_git / 'data' / 'Lexplore_metadata.yaml'
with open(path_to_config, 'r') as config_file:
    cfg_metadata = yaml.safe_load(config_file)
if 'ecotaxa_initial_classification_id' in cfg_metadata.keys():
    cfg_metadata['ecotaxa_initial_classification_id']=eval(cfg_metadata['ecotaxa_initial_classification_id'])
path_to_network=Path("{}:{}".format(cfg_metadata['local_network'],os.path.sep)) # Set working directory to forel-meco
path_to_ecotaxa_cytosense_files=Path(path_to_network /'lexplore' / 'LeXPLORE' / 'ecotaxa'  )
path_to_ecotaxa_flowcam_files=Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data'  / 'Lexplore' / 'ecotaxa'  )
df_context_flowcam_micro=pd.read_csv(r'{}'.format(cfg_metadata['flowcam_10x_context_file']),sep=r'\=|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains(r"\[",case=True)').set_index('Name').T.rename(index={'Value':Path(cfg_metadata['flowcam_10x_context_file']).stem})
df_context_flowcam_macro=pd.read_csv(r'{}'.format(cfg_metadata['flowcam_macro_context_file']),sep=r'\=|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains(r"\[",case=True)').set_index('Name').T.rename(index={'Value':Path(cfg_metadata['flowcam_macro_context_file']).stem})
# Remove duplicated columns
df_context_flowcam_macro=df_context_flowcam_macro.loc[:,~df_context_flowcam_macro.columns.duplicated()].copy()
df_context_flowcam_micro=df_context_flowcam_micro.loc[:,~df_context_flowcam_micro.columns.duplicated()].copy()
# Merge in a single context file
df_context=pd.concat([pd.concat([df_context_flowcam_micro,pd.DataFrame(dict(zip([column for column in df_context_flowcam_macro.columns if column not in df_context_flowcam_micro.columns],[pd.NA]*len([column for column in df_context_flowcam_macro.columns if column not in df_context_flowcam_micro.columns]))),index=df_context_flowcam_micro.index)],axis=1),pd.concat([df_context_flowcam_macro,pd.DataFrame(dict(zip([column for column in df_context_flowcam_micro.columns if column not in df_context_flowcam_macro.columns],[pd.NA]*len([column for column in df_context_flowcam_micro.columns if column not in df_context_flowcam_macro.columns]))),index=df_context_flowcam_macro.index)],axis=1)],axis=0)

def resize_to_square(image, size):
    h, w, d = image.shape
    ratio = size / max(h, w)
    resized_image = cv2.resize(image, (int( max(h, w)*ratio), int( max(h, w)*ratio)), cv2.INTER_AREA)
    return resized_image

def image_to_tensor(image, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(image / (255. if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor

def pad(image, min_height, min_width):
    h,w,d = image.shape

    if h < min_height:
        h_pad_top = int((min_height - h) / 2.0)
        h_pad_bottom = min_height - h - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if w < min_width:
        w_pad_left = int((min_width - w) / 2.0)
        w_pad_right = min_width - w - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    return cv2.copyMakeBorder(image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))


class image_Dataset(torch.utils.data.Dataset):

    def __init__(self, df, size,scale_value,pixel_size):
        self.df = df
        self.size = size
        self.scale_value=scale_value
        self.pixel_size=pixel_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        padding = int((np.ceil(self.scale_value / self.pixel_size) + 10) / 2)
        image = cv2.imread(row.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = pad(resize_to_square(image, size=np.max(image.shape)), int(self.size / self.pixel_size), int(self.size / self.pixel_size))
        tensor = image_to_tensor(image, normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
        # tensor = np.transpose(tensor.numpy(),(1,2,0))
        # plt.figure(),plt.imshow(tensor, cmap='gray'),plt.show()

        return tensor

def properties_to_thumbnails(image, df_properties, save_directory):
        """
        Objective: This function generates thumbnails of region of interest based on the properties returned by image processing
        :param image: The initial image from which regions of interest have been segmented
        :param df_properties: A dataframe including two feature-based axis ('x' and 'y') along the local path of the corresponding images ('image_url')
        :return: Returns the dash application
        """
        # scale_value = 300 if '_'.join(save_directory.parent.stem.split('_')[0:2]).lower() == 'flowcam_2m' else 50  # size of the scale bar in microns
        pixel_size = ((df_properties.area / df_properties.num_pixels) ** 0.5).values[0]
        # padding = int((np.ceil(scale_value / pixel_size) + 10) / 2)

        for particle_id in df_properties.index.astype(str):
            thumbnail = Image.fromarray(image[tuple(
                map(lambda s: slice(np.max([0, s[1].start - 10]), np.min([s[1].stop + 10, image.shape[s[0]]]), None),
                    enumerate(list(df_properties.at[int(particle_id), 'slice']))))][
                                            slice(1, -1, None), slice(1, -1, None)])
            # thumbnail.show()
            save_directory.mkdir(exist_ok=True)
            thumbnail.save(str(save_directory / 'thumbnail_particle_{}.jpg'.format(str(particle_id).rstrip())))

def scatter_2d_images(df_directory,axis_dict):
        """
        Objective: This function returns an interactive scatter plot rendering local images on hovering near a feature datapoint
        :param df: A dataframe including two feature-based axis ('x' and 'y') along the local path of the corresponding images ('image_url')
        :return: Returns the dash application
        """

        df=pd.read_csv(r'{}'.format(df_directory)).rename(columns=axis_dict)
        df=df.dropna(subset=['images'])
        dbscan = DBSCAN(eps=0.95, min_samples=5,leaf_size=12)
        cluster=dbscan.fit(df[['x','y']].to_numpy())
        lof = LocalOutlierFactor(n_neighbors=10)
        lof.fit_predict(df[['x','y']].to_numpy())
        df['score']=lof.negative_outlier_factor_
        df['cluster']=cluster.labels_
        df=df.assign(color=lambda x: x.cluster.astype(str).isin(['0']), size=lambda x: np.where(x.score < -1.7, 30, 4))

        df['color'] = df.color if 'color' in df.columns else 'rgb(0,0,0)'
        df['size'] = df['size'] if 'size' in df.columns else 5
        df['img_url'] = df.images.apply(lambda image_source: html.Img("{}{}".format("data:image/jpg;base64, ", base64.b64encode(open(image_source,'rb').read()).decode('utf-8'))).to_plotly_json())

        fig = px.scatter(df, x="x", y="y", custom_data=["images"], color='color', size='size', hover_data='images')

        # Update layout and update traces
        fig.update_layout(clickmode='event+select', xaxis=dict(showgrid=False, showline=True, linecolor='black'),
                          yaxis=dict(showgrid=False, showline=True, linecolor='black'),
                          plot_bgcolor='rgba(0,0,0,0)')  # type="log",
        fig.update_traces(hoverinfo="none", hovertemplate=None)
        app = Dash()

        app.layout = html.Div(
            [
                dcc.Graph(id="graph_interaction", figure=fig, clear_on_unhover=True, style={'height': '90vh'}),
                dcc.Tooltip(id="graph-tooltip"),

            ]
        )

        @app.callback(Output("graph-tooltip", "show"),Output("graph-tooltip", "bbox"),Output('graph-tooltip', 'children'),Input('graph_interaction', 'hoverData'))
        def open_url(hoverData):
            if hoverData:
                pt = hoverData["points"][0]
                bbox = pt["bbox"]
                id = pt["pointIndex"]
                image_source = pt['customdata'][0]  # df.loc[id,'images']
                if str(image_source) != 'nan':
                    img_data = base64.b64encode(open(image_source, 'rb').read()).decode('utf-8')
                    image_url = "{}{}".format("data:image/jpg;base64, ", img_data)
                    children = [html.Div([html.H6( 'Particle ID: {}'.format(image_source.split(os.sep)[-1].split('_')[-1].replace('.jpg',''))),html.Img(src=image_url, style={"width": "100%"})],
                                         style={'width': '100px', 'white-space': 'normal'})]

                    return True, bbox, children
            else:
                raise PreventUpdate

        return app,df

#image_path=r"R:\Imaging_Flowcam\Flowcam data\Lexplore\cnn\input\Flowcam_2mm_lexplore_wasam_20250115_2025-01-16"
def images_to_dataset(image_path,filter,image_extension='.jpg'):
    images=list(Path(image_path).expanduser().rglob('*{}'.format(image_extension)))
    if len(filter):
        images=[image for image in images if image.stem+'.jpg' in filter.tolist()]
    df_path=pd.DataFrame({'ID':[path.stem for path in images],'path':[str(path) for path in images]})
    df_context.loc['context_flowcam_2mm_lexplore','pixel_size']=cfg_metadata['pixel_size_flowcam_macro']
    df_context.loc['context_flowcam_10x_lexplore', 'pixel_size'] = cfg_metadata['pixel_size_flowcam_10x']
    df_context.loc['context_cytosense_lexplore', 'pixel_size'] = cfg_metadata['pixel_size_cytosense']
    df_context.loc['context_cytosense_lexplore', ['AcceptableTop','AcceptableLeft','AcceptableBottom','AcceptableRight']] = 0,0,cfg_metadata['height_cytosense'],cfg_metadata['width_cytosense']
    id_instrument=[parent in str(Path(image_path)) for parent in [str(Path(r'{}\\Flowcam_10x'.format(Path(image_path).parent))),str(Path(r'{}\\Flowcam_2mm'.format(Path(image_path).parent))),str(Path(r"R:\lexplore\LeXPLORE\ecotaxa"))]]
    pixel_size=df_context.loc[id_instrument, 'pixel_size'].values[0]
    max_size = int(np.max([(df_context.loc[id_instrument].AcceptableBottom.astype(float) - df_context.loc[ id_instrument].AcceptableTop.astype(float)) * pixel_size, (df_context.loc[id_instrument].AcceptableRight.astype(float) - df_context.loc[ id_instrument].AcceptableLeft.astype(float)) * pixel_size]))  # int(np.max([(df_context.AcceptableBottom.astype(float)-df_context.AcceptableTop.astype(float))*df_context.pixel_size,(df_context.AcceptableRight.astype(float)-df_context.AcceptableLeft.astype(float))*df_context.pixel_size]))

    scale_value = 300 if '_'.join(Path(image_path).stem.lower().split('_')[0:2]) == 'flowcam_2mm' else 50  # size of the scale bar in microns

    return image_Dataset(df=df_path, size=max_size,scale_value=scale_value,pixel_size=pixel_size)



def image_feature_dataset(image_path,filter,model_name='resnet18',layer_name = 'avgpool'):
    model = getattr(torchvision.models, model_name)(pretrained=True)
    layer=model._modules.get(layer_name)

    model.eval()
    return_nodes = {'flatten': 'flatten'}
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    '''
    features=torch.zeros(512)
    features = []
    def hook_features(module, input, output):
        N, C, H, W = output.shape
        output = output.reshape(N, C, -1)
        features.append(output.mean(dim=2).cpu().detach().numpy())
        return features
    handle = model._modules.get(layer_name).register_forward_hook(hook_features)
    '''
    dataset =images_to_dataset(image_path,filter,image_extension='.jpg')
    loader = torch.utils.data.DataLoader(dataset,  batch_size=1, shuffle=False, num_workers=1)
    df_features=pd.concat([dataset.df,pd.DataFrame('feature_'+pd.Series((np.arange(1,513))).astype(str)).set_index(0).T],axis=1)
    for i_batch, inputs in tqdm(enumerate(loader), total=len(loader)):
        df_features.loc[i_batch,df_features.columns[np.arange(2,df_features.shape[1])]]=feature_extractor(inputs)['flatten'].squeeze().tolist()

    del model

    return df_features

if __name__ == '__main__':
    path = sys.argv[1]
    x_axis_dict=str(sys.argv[2])
    y_axis_dict = str(sys.argv[3])

    app,clusters=scatter_2d_images(df_directory=path,axis_dict={x_axis_dict:"x",y_axis_dict:"y"})
    app.run_server(debug=True, use_reloader=False)