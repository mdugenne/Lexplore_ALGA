import cv2
import torch #pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
import imagesize
import numpy  as np
import pandas as pd
from torchvision.models import efficientnet
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
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from itertools import compress
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
# silhouette_score for evaluating the clusters, ranges from -1 to +1, desirable score is close to 1
from sklearn.metrics import silhouette_score


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
if Path(r'{}'.format(cfg_metadata['flowcam_10x_context_file'])).expanduser().exists():
    df_context_flowcam_micro=pd.read_csv(r'{}'.format(cfg_metadata['flowcam_10x_context_file']),sep=r'\=|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains(r"\[",case=True)').set_index('Name').T.rename(index={'Value':Path(cfg_metadata['flowcam_10x_context_file']).stem})
    df_context_flowcam_macro=pd.read_csv(r'{}'.format(cfg_metadata['flowcam_macro_context_file']),sep=r'\=|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains(r"\[",case=True)').set_index('Name').T.rename(index={'Value':Path(cfg_metadata['flowcam_macro_context_file']).stem})
    # Remove duplicated columns
    df_context_flowcam_macro=df_context_flowcam_macro.loc[:,~df_context_flowcam_macro.columns.duplicated()].copy()
    df_context_flowcam_micro=df_context_flowcam_micro.loc[:,~df_context_flowcam_micro.columns.duplicated()].copy()
    # Merge in a single context file
    df_context=pd.concat([pd.concat([df_context_flowcam_micro,pd.DataFrame(dict(zip([column for column in df_context_flowcam_macro.columns if column not in df_context_flowcam_micro.columns],[pd.NA]*len([column for column in df_context_flowcam_macro.columns if column not in df_context_flowcam_micro.columns]))),index=df_context_flowcam_micro.index)],axis=1),pd.concat([df_context_flowcam_macro,pd.DataFrame(dict(zip([column for column in df_context_flowcam_micro.columns if column not in df_context_flowcam_macro.columns],[pd.NA]*len([column for column in df_context_flowcam_micro.columns if column not in df_context_flowcam_macro.columns]))),index=df_context_flowcam_macro.index)],axis=1)],axis=0)
else:
    df_context = pd.DataFrame({'pixel_size': [9.47,0.7339,0.67]}, index=['context_flowcam_2mm_lexplore', 'context_flowcam_10x_lexplore', 'context_cytosense_lexplore'])
    df_context.loc['context_flowcam_2mm_lexplore', ['AcceptableTop', 'AcceptableLeft', 'AcceptableBottom', 'AcceptableRight']] = 10, 80, 1850, 1125
    df_context.loc['context_flowcam_10x_lexplore', ['AcceptableTop', 'AcceptableLeft', 'AcceptableBottom', 'AcceptableRight']] = 1, 115, 1919, 1085
    df_context.loc['context_cytosense_lexplore', ['AcceptableTop', 'AcceptableLeft', 'AcceptableBottom','AcceptableRight']] = 0, 0, 1024, 1280

dict_properties_visual_spreadsheet={'Name':['[t]'], 'Area (ABD)':['[f]'], 'Area (Filled)':['[f]'], 'Aspect Ratio':['[f]'], 'Average Blue':['[f]'],
       'Average Green':['[f]'], 'Average Red':['[f]'], 'Calibration Factor':['[f]'],
       'Calibration Image':['[f]'], 'Capture ID':['[t]'] ,'Capture X':['[f]'], 'Capture Y':['[f]'], 'Circularity':['[f]'],
       'Circularity (Hu)':['[f]'], 'Compactness':['[f]'], 'Convex Perimeter':['[f]'], 'Convexity':['[f]'], 'Diameter (ABD)':['[f]'], 'Diameter (ESD)':['[f]'], 'Diameter (FD)':['[f]'],
       'Edge Gradient':['[f]'], 'Elapsed Time':['[f]'], 'Elongation':['[f]'], 'Feret Angle Max':['[f]'],
       'Feret Angle Min':['[f]'], 'Fiber Curl':['[f]'], 'Fiber Straightness':['[f]'], 'Filter Score':['[f]'],
       'Geodesic Aspect Ratio':['[f]'], 'Geodesic Length':['[f]'], 'Geodesic Thickness':['[f]'],
       'Group ID':['[t]'], 'Image Height':['[f]'], 'Image Width':['[f]'], 'Image X':['[f]'], 'Image Y':['[f]'],
       'Length':['[f]'], 'Particles Per Chain':['[f]'], 'Perimeter':['[f]'], 'Ratio Blue/Green':['[f]'],
       'Ratio Red/Blue':['[f]'], 'Ratio Red/Green':['[f]'], 'Roughness':['[f]'], 'Source Image':['[f]'],
       'Sum Intensity':['[f]'], 'Symmetry':['[f]'],  'Transparency':['[f]'], 'Volume (ABD)':['[f]'],
       'Width':['[f]']}
#Define a metadata table for Lexplore based on the metadata confiugration file
df_sample=pd.DataFrame({'sample_longitude':['[f]',0],'sample_latitude':['[f]',0],'sample_platform':['[t]','lab'],'sample_project':['[t]','sinking'],'sample_type':['[t]','culture'] })
df_acquisition=pd.DataFrame({'acq_instrument':['[t]','Flowcam_8000'],'acq_pumptype':['[t]','syringe'],'acq_pixel_um':['[f]',cfg_metadata['pixel_size_flowcam_10x']],'acq_frequency_fps':['[f]',1],'acq_max_width_um':['[f]',100],'acq_max_height_um':['[f]',300]})
df_processing=pd.DataFrame({'process_code':['[t]','https://github.com/mdugenne/Lexplore_ALGA/scripts/script_flowcam_sinking'],'process_min_diameter_um':['[f]',10],'process_max_diameter_um':['[f]',100]})

import plotnine
from plotnine import geom_errorbar, coord_fixed, scale_fill_distiller
from plotnine.themes.themeable import legend_position

theme_plot=theme(panel_grid=element_blank(), legend_position='bottom',
              panel_background=element_rect(fill='#ffffff'), legend_background=element_rect(fill='#ffffff'),
              strip_background=element_rect(fill='#ffffff'),
              panel_border=element_rect(color='#222222'),
              legend_title=element_text(family='Times New Roman', size=12),
              legend_text=element_text(family='Times New Roman', size=12),
              axis_title=element_text(family='Times New Roman', size=12, linespacing=1),
              axis_text_x=element_text(family='Times New Roman', size=12, linespacing=0),
              axis_text_y=element_text(family='Times New Roman', size=12, rotation=90, linespacing=1),
              plot_background=element_rect(fill='#ffffff00'))


def generate_ecotaxa_table(df,instrument,path_to_storage=None):
    """
    Objective: This function generates a table to be uploaded on Ecotaxa along with the thumbnails after acquisitions with the CytoSense or FlowCam imaging flow cytometer
    :param df: A dataframe of morphometric properties including *area* estimates for individual particles. \nFor FlowCam acquisitions, this tables corresponds to the "data" csv file that can be exported automatically (in context file, go to the Reports tab and enable option "Export list data when run terminates") or manually (in the main window, click on the File tab and select "Export data". Note that if the run was closed you'll need to re-open the data by clicking on "Open data" amd selecting the run).
    \nFor CytoSense acquisitions, the table is generated by running the custom 0_upload_thumbnails_cytosense.py. This script uses cropped images (without scalebars) exported automatically after each acquisition.
    :param instrument: Instrument used for the acquisition. Supported options are "CytoSense" or "FlowCam"
    :param path_to_storage: A Path object specifying where the table should be saved. If None, the function returns the table that is not saved.
    :return: Returns a dataframe including with the appropriate format for upload on Ecotaxa. See instructions at https://ecotaxa.obs-vlfr.fr/gui/prj/
    """

    df_ecotaxa_object=pd.concat([pd.DataFrame(
        {'img_file_name': ['[t]'] + list(df.img_file_name.values),
         'object_id': ['[t]'] + list(df.img_file_name.astype(str)), # Attention: The object_id needs to include the file extension
         'object_lat': ['[f]'] + [0] * len(df),
         'object_lon': ['[f]'] + [0] * len(df),
         'object_date': ['[t]'] + [df.Name.astype(str).values[0].split('_')[5].replace('-','')[0:8]] * len(df),
         'object_time': ['[t]'] + ['000000'] * len(df),
         'object_depth_min': ['[f]'] + [0] * len(df),
         'object_depth_max': ['[f]'] + [0] * len(df)
         }),pd.concat([pd.DataFrame(dict(zip(('object_'+pd.Series([key.replace(' ','_').replace('/','_').replace('(','').replace(')','').lower() for key in {**dict_properties_visual_spreadsheet}.keys() if key in df.columns])).values,[value for key,value in {**dict_properties_visual_spreadsheet}.items() if key in df.columns]))),df.rename(columns=dict(zip([column for column in df.columns if column in {**dict_properties_visual_spreadsheet}.keys()],['object_'+column.replace(' ','_').replace('/','_').replace('(','').replace(')','').lower() for column in df.columns if column in {**dict_properties_visual_spreadsheet}.keys()])))[['object_'+column.replace(' ','_').replace('/','_').replace('(','').replace(')','').lower() for column in df.columns if column in {**dict_properties_visual_spreadsheet}.keys()]]],axis=0).reset_index(drop=True)],axis=1).reset_index(drop=True)

    df_ecotaxa_sample=pd.concat([pd.DataFrame({'sample_id': ['[t]'] + list(df.Name.astype(str).values)}),pd.concat([df_sample,*[df_sample.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True)],axis=1)
    df_acq=df_acquisition
    df_ecotaxa_acquisition=pd.concat([df_acq,*[df_acq.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True).assign(acq_id=['[t]'] +list(df.Name.astype(str).values))
    df_process = df_processing
    df_ecotaxa_process=pd.concat([df_process,*[df_process.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True).assign(process_id=['[t]'] +['process_SickNSinking']*len(df))
    df_ecotaxa=pd.concat([df_ecotaxa_object,df_ecotaxa_sample,df_ecotaxa_acquisition[['acq_id']+list(df_ecotaxa_acquisition.columns[0:-1])],df_ecotaxa_process[['process_id']+list(df_ecotaxa_process.columns[0:-1])]],axis=1)


    if len(path_to_storage):
        df_ecotaxa.to_csv(path_to_storage, sep='\t',index=False)
    return df_ecotaxa

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

def scatter_2d_images(df_directory):
        """
        Objective: This function returns an interactive scatter plot rendering local images on hovering near a feature datapoint
        :param df: A dataframe including two feature-based axis ('x' and 'y') along the local path of the corresponding images ('image_url')
        :return: Returns the dash application
        """

        df=pd.read_csv(r'{}'.format(df_directory)).rename(columns={'0': 'x', '1': 'y','2':'z'})#.rename(columns=axis_dict)
        nn = NearestNeighbors(n_neighbors=20).fit(df[['x','y','z']])
        distances, indices = nn.kneighbors(df[['x','y','z']])
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        min_samples = range(1, 51)
        eps = np.arange(np.quantile(distances,q=0.05), 2*np.quantile(distances,q=1), 0.1)  # returns array of ranging from 0.05 to 0.13 with step of 0.01
        output = []
        for ms in min_samples:
            for ep in eps:
                labels = DBSCAN(min_samples=ms, eps=ep).fit(df[['x','y','z']]).labels_

                score =-1 if len(np.unique(labels))==1 else silhouette_score(df[['x','y','z']], labels)
                output.append((ms, ep, score))
        min_samples, eps, score = sorted(output, key=lambda x: x[-1])[-1]
        dbscan = DBSCAN(eps=eps,min_samples=min_samples)
        cluster=dbscan.fit(df[['x','y','z']].to_numpy())
        euclidean_distance=sklearn.metrics.pairwise.euclidean_distances(df[['x','y','z']])
        euclidean_distance[np.where(euclidean_distance==0)]=np.inf
        #euclidean_distance=np.where(np.triu(euclidean_distance) == 0, np.inf, euclidean_distance)

        # Replace outlier cluster label by nearest neighbor's
        #itemgetter(*list(itertools.combinations(list(np.where(cluster.labels_ == -1)[0]), 2)))( euclidean_distance)
        for i in list(np.where(cluster.labels_ == -1)[0]):
            euclidean_distance[i][list(np.where(cluster.labels_ == -1)[0])]=np.inf
            euclidean_distance[:,i][list(np.where(cluster.labels_ == -1)[0])] = np.inf

        closest_neighbor=np.argmin(euclidean_distance, axis=0)
        for i in list(np.where(cluster.labels_ == -1)[0]):
            cluster.labels_[i]=cluster.labels_[closest_neighbor[i]]


        lof = LocalOutlierFactor(n_neighbors=10)
        lof.fit_predict(df[['x','y']].to_numpy())
        df['score']=lof.negative_outlier_factor_
        df['cluster']=cluster.labels_
        df=df.assign(color=lambda x: x.cluster.astype(str), size=lambda x: np.where(x.score < -1.7, 30, 4))

        df['color'] = df.color if 'color' in df.columns else 'rgb(0,0,0)'
        df['size'] = df['size'] if 'size' in df.columns else 5
        df['img_url'] = df.images.apply(lambda image_source: html.Img("{}{}".format("data:image/jpg;base64, ", base64.b64encode(open(image_source,'rb').read()).decode('utf-8'))).to_plotly_json())

        #idx=np.random.randint(df.cluster.nunique(),size=1)
        #df=df_all.loc[df_all.cluster==idx[0]]
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
                    children = [html.Div([html.Img(src=image_url, style={"width": "400%"}),html.H6( 'Particle ID: {}'.format(image_source.split(os.sep)[-1].split('_')[-1].replace('.jpg','')))],
                                         style={'width': '100px', 'white-space': 'normal'})]

                    return True, bbox, children
            else:
                raise PreventUpdate

        return app,df
import scipy.stats as stats
#image_path=r"R:\Imaging_Flowcam\Flowcam data\Lexplore\cnn\input\Flowcam_2mm_lexplore_wasam_20250115_2025-01-16"
def images_to_dataset(image_path,filter,image_extension='.jpg'):
    images=list(Path(image_path).expanduser().rglob('*{}'.format(image_extension)))
    if len(filter):
        images=[image for image in images if image.stem+'.jpg' in filter.tolist()]
    df_path=pd.DataFrame({'ID':[path.stem for path in images],'path':[str(path) for path in images]})
    id_instrument='flowcam'
    pixel_size=0.7339
    max_size = int(np.max([(1918 - 1) * pixel_size, (1015 - 185) * pixel_size]))  # int(np.max([(df_context.AcceptableBottom.astype(float)-df_context.AcceptableTop.astype(float))*df_context.pixel_size,(df_context.AcceptableRight.astype(float)-df_context.AcceptableLeft.astype(float))*df_context.pixel_size]))

    scale_value =  50  # size of the scale bar in microns

    return image_Dataset(df=df_path, size=max_size,scale_value=scale_value,pixel_size=pixel_size)



def image_feature_dataset(image_path,filter,model_name='efficientnet_b0',layer_name = 'avgpool'): #resnet18,512
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
    df_features=pd.concat([dataset.df,pd.DataFrame('feature_'+pd.Series((np.arange(1,1280+1))).astype(str)).set_index(0).T],axis=1)
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