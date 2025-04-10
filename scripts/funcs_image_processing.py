## Objective: This script includes all modules and functions required for image processing

import warnings
warnings.filterwarnings(action='ignore')

# Image processing modules
import skimage as ski #pip install -U scikit-image
from skimage import color,measure,morphology
from PIL import Image
from skimage.util import compare_images,crop
from skimage.color import rgb2gray
from skimage.measure import regionprops
import cv2
import torch
from scripts.funcs_image_utils import *

# Data processing modules
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from itertools import compress
from sklearn.neighbors import NearestNeighbors

# Web scraping:
import urllib3
import re
import string
import requests
from bs4 import BeautifulSoup
import urllib
from urllib.parse import urljoin
from requests_html import HTMLSession #pip install requests_html, lxml_html_clean

# Plot modules
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
fontprops = fm.FontProperties(size=14,family='serif')
import copy
my_cmap = copy.copy(plt.colormaps.get_cmap('gray_r')) # get a copy of the gray color map
my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values
import latex

## interactive module
import dash
from dash.exceptions import PreventUpdate
from dash import Dash,dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import base64

from plotnine import *
import plotnine
import matplotlib
from plotnine import geom_errorbar, coord_fixed, scale_fill_distiller
from plotnine.themes.themeable import legend_position

theme_paper=theme(panel_grid=element_blank(), legend_position='bottom',
              panel_background=element_rect(fill='#ffffff'), legend_background=element_rect(fill='#ffffff'),
              strip_background=element_rect(fill='#ffffff'),
              panel_border=element_rect(color='#222222'),
              legend_title=element_text(family='Times New Roman', size=12),
              legend_text=element_text(family='Times New Roman', size=12),
              axis_title=element_text(family='Times New Roman', size=12, linespacing=1),
              axis_text_x=element_text(family='Times New Roman', size=12, linespacing=0),
              axis_text_y=element_text(family='Times New Roman', size=12, rotation=90, linespacing=1),
              plot_background=element_rect(fill='#ffffff00'))

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
dict_ecotaxa_types={'img_file_name':str,'object_id':str,'object_lat':float,'object_lon':float,'object_date':str,'object_time':str,'object_depth_min':float,'object_depth_max':float,'object_annotation_category':str,'object_annotation_status':str,'object_annotation_date':str,'object_annotation_time':str,'object_annotation_person_name':str,'object_annotation_person_email':str,'sample_id':str,'sample_latitude':float,'sample_longitude':float,'sample_platform':str,'sample_project':str,'sample_principal_investigator':str,'sample_depthprofile':str,'sample_storage_gear':str,'sample_fixative':str,'sample_fixative_final_concentration':float,'sample_volume_analyzed_ml':float,'sample_volume_pumped_ml':float,'sample_volume_fluid_imaged_ml':float,'sample_duration_sec':str,'acq_id':str,'acq_instrument':str,'acq_software':str,'acq_flowcell':str,'acq_objective':str,'acq_mode':str,'acq_stop_criterion':str,'acq_pumptype':str,'acq_max_width_um':float,'acq_max_height_um':float,'acq_min_diameter_um':float,'acq_max_diameter_um':float,'acq_dark_threshold':float,'acq_light_threshold':float,'acq_neighbor_distance':float,'acq_closing_iterations':float,'acq_rolling_calibration':float,'acq_flow_rate':float,'process_id':str,'process_operator':str,'process_code':str,'process_imagetype':str}

import shutil # zip folder
import re
import time,datetime
path_to_config_logon = path_to_git / 'data' / 'datafiles' / 'Lexplore_configuration_logon.yaml'
with open(path_to_config_logon, 'r') as config_file:
    cfg_logon = yaml.safe_load(config_file)
import itertools

# Ecotaxa API module
## See documentation at https://github.com/ecotaxa/ecotaxa_py_client
## Use pip install git+https://github.com/ecotaxa/ecotaxa_py_client.git
import ecotaxa_py_client
from ecotaxa_py_client import AuthentificationApi
from ecotaxa_py_client import LoginReq
from ecotaxa_py_client.models.create_project_req import CreateProjectReq
from ecotaxa_py_client.models.min_user_model import MinUserModel
from ecotaxa_py_client.models.project_model import ProjectModel
from ecotaxa_py_client.models.import_req import ImportReq
from ecotaxa_py_client.models.taxon_model import TaxonModel
from ecotaxa_py_client.models.user_model_with_rights import UserModelWithRights
from ecotaxa_py_client.api import objects_api
from ecotaxa_py_client.models.body_export_object_set_object_set_export_post import BodyExportObjectSetObjectSetExportPost
from ecotaxa_py_client.models.project_filters import ProjectFilters
from ecotaxa_py_client.models.historical_classification import HistoricalClassification
from ecotaxa_py_client.models.export_req import ExportReq
from ecotaxa_py_client.api import jobs_api

## Step 1: Create an API instance based on authentication infos.
with ecotaxa_py_client.ApiClient() as client:
    api = AuthentificationApi(client)
    token = api.login(LoginReq(username=cfg_logon['login_ecotaxa'], password=cfg_logon['password_ecotaxa']))
configuration = ecotaxa_py_client.Configuration(host="https://ecotaxa.obs-vlfr.fr/api", access_token=token)
configuration.verify_ssl = False

# create_ecotaxa_user()
def create_ecotaxa_user(ecotaxa_configuration=ecotaxa_py_client.Configuration(host="https://ecotaxa.obs-vlfr.fr/api"),user_config={'name':cfg_metadata['principal_investigator'],'email':'Bastiaan.Ibelings@unige.ch'}):
    """
      Objective: This function uses Ecotaxa (https://ecotaxa.obs-vlfr.fr) programming interface to create a new account on Ecotaxa based on credentials
      Documentation available at : https://github.com/ecotaxa/ecotaxa_py_client/blob/main/docs/ProjectsApi.md#create_user

      :param configuration: An API instance specifying the host (Ecotaxa url) and the token generated based on Ecotaxa credentials. Attention: Logged users cannot create another account
      :param project_config: A dictionary containing all the info necessary to create a new project on Ecotaxa. \nOptions include:
      clone_of_id (integer): Existing project ID used as a template for the new project (taxonomy will be copied). Requires appropriate access rights\n
      title (string): The name of the new project\n
      instrument (string): The instrument name. Current options include AMNIS, CPICS, CytoSense, FastCam, FlowCam, IFCB, ISIIS, LISST-Holo, Loki, Other camera, Other flowcytometer, Other microscope, Other scanner, PlanktoScope, Plankton Imager, UVP5HD, UVP5SD, UVP5Z, UVP6, VPR, ZooCam, Zooscan, eHFCM\n
      :return: Returns user ID
    """
    print('To register on Ecotaxa, please follow the instructions at: https://ecotaxa.obs-vlfr.fr/gui/register/?.\nAn account cannot be created through the API unless you are an administrator of Ecotaxa.')
    return None
    with ecotaxa_py_client.ApiClient(ecotaxa_configuration) as api_client:
        # Create an instance of the API class
        api_instance = ecotaxa_py_client.UsersApi(api_client)
        user_model_with_rights = ecotaxa_py_client.models.MinimalUserBO(name=user_config['name'])  # UserModelWithRights |
        no_bot = ["['127.0.0.1', 'ffqsdfsdf']"]  # List[str] | not-a-robot proof (optional)
        token = 'token_example'  # str | token in the url to validate request (optional)

        try:
            # Create User
            api_response = api_instance.create_user(user_model_with_rights, no_bot=no_bot, token=token)
            print("The response of UsersApi->create_user:\n")
            print(api_response)
        except Exception as e:
            print("Exception when calling UsersApi->create_user: %s\n" % e)


def create_ecotaxa_project(ecotaxa_configuration=configuration,project_config={'clone of id':14791,'title':'Lexplore_ALGA_Flowcam_micro','instrument':'FlowCam','managers':['Bas Ibelings','Irene Muscas','Mathilde Dugenne'],'project_description':'This dataset includes thumbnails generated by a FlowCam micro. Acquisitions are done on samples collected on a daily basis at the surface of Lake Geneva as part of the ALGA project (PI: Bastiaan Ibelings)','project_sorting_fields':r'area=Area [squared micrometers]\r\nnb_particles=Number of particles in the foreground\r\naxis_major_length= Major ellipsoidal axis [micrometers]\r\nequivalent_diameter_area= Equivalent circular diameter [micrometers]\r\nvisualspreadsheet_circularity=Circularity\r\nvisualspreadsheet_average_blue=Average blue intensity\r\nvisualspreadsheet_average_green=Average green intensity\r\nvisualspreadsheet_average_red=Average Red intensity\r\nvisualspreadsheet_edge_gradient=Edge gradient\r\nvisualspreadsheet_elongation=Elongation\r\nvisualspreadsheet_ratio_blue_green=Blue green ratio\r\nvisualspreadsheet_ratio_red_blue=Red blue ratio\r\nvisualspreadsheet_ratio_red_green=Red green ratio\r\nvisualspreadsheet_transparency=Transparency'},update_configuration=True):
    """
     Objective: This function uses Ecotaxa (https://ecotaxa.obs-vlfr.fr) programming interface to create a new project on Ecotaxa based on account credentials
     Documentation available at : https://github.com/ecotaxa/ecotaxa_py_client/blob/main/docs/ProjectsApi.md#create_project

     :param configuration: An API instance specifying the host (Ecotaxa url) and the token generated based on Ecotaxa credentials
     :param project_config: A dictionary containing all the info necessary to create a new project on Ecotaxa. \nOptions include:
     clone_of_id (integer): Existing project ID used as a template for the new project (taxonomy will be copied). Requires appropriate access rights\n
     title (string): The name of the new project\n
     instrument (string): The instrument name. Current options include AMNIS, CPICS, CytoSense, FastCam, FlowCam, IFCB, ISIIS, LISST-Holo, Loki, Other camera, Other flowcytometer, Other microscope, Other scanner, PlanktoScope, Plankton Imager, UVP5HD, UVP5SD, UVP5Z, UVP6, VPR, ZooCam, Zooscan, eHFCM\n
     :return: Returns new project ID
     """
    id_to_clone=int(project_config['clone of id']) if 'clone of id' in project_config.keys() else None
    title = project_config['title'] if 'title' in project_config.keys() else 'test_project'
    instrument = project_config['instrument'] if 'instrument' in project_config.keys() else 'Other camera'
    description = project_config['project_description'] if 'project_description' in project_config.keys() else ''
    sorting_columns=project_config['project_sorting_fields'] if 'project_sorting_fields' in project_config.keys() else ''
    # Step 1 : Create the new project according to the info in project_config
    with ecotaxa_py_client.ApiClient(ecotaxa_configuration) as api_client:
        # Create an instance of the API class
        api_instance = ecotaxa_py_client.ProjectsApi(api_client)
        create_project_req = ecotaxa_py_client.CreateProjectReq(clone_of_id=id_to_clone,title=title,instrument=instrument,visible=True)  # CreateProjectReq |
        api_instance_users = ecotaxa_py_client.UsersApi(api_client)
        api_instance_taxo = ecotaxa_py_client.TaxonomyTreeApi(api_client)
        api_response_user = list(map(lambda user: api_instance_users.search_user(by_name=user)[-1],project_config['managers']))
        try:
            # Step 1:Create a new Project and save ID in the configuration file
            api_response = api_instance.create_project(create_project_req)
            project_id=api_response
            if ('ecotaxa_{}_projectid'.format(title.lower().replace(' ','_')) not in cfg_metadata.keys()) & (update_configuration):
                with open(path_to_config, 'w') as config_file:
                    if 'ecotaxa_initial_classification_id' in cfg_metadata.keys():
                        if type(cfg_metadata['ecotaxa_initial_classification_id']) is list:
                            cfg_metadata['ecotaxa_initial_classification_id'] = str(cfg_metadata['ecotaxa_initial_classification_id'])
                    cfg_metadata.update({'ecotaxa_{}_projectid'.format(title.lower().replace(' ','_')): project_id})
                    yaml.dump(cfg_metadata, config_file, default_flow_style=False)


            # Step 2: Create a project model based on existing project (for taxonomy)
            if id_to_clone!=None:
                project_model_clone = api_instance.project_query(project_id=id_to_clone, for_managing=False)
                # Retrieve list of taxa and sorting columns from cloned project
                initial_classification_list = project_model_clone.init_classif_list

                if ('ecotaxa_initial_classification_id' not in cfg_metadata.keys()) & (update_configuration):
                    with open(path_to_config, 'w') as config_file:
                        cfg_metadata.update({'ecotaxa_initial_classification_id':str(initial_classification_list)})
                        yaml.dump(cfg_metadata,config_file, default_flow_style=False)

                    request=input('Creating an Ecotaxa project from scratch results in empty taxonomy selection.\nYou may add categories of interest at a later stage or look for existing categories that would match your IDs.\nType 1 to continue without an initial selection, 2 to attempt finding a match up based on known categories, or press enter to use categories stored in default project ({})'.format('https://ecotaxa.obs-vlfr.fr/prj/'+str(id_to_clone)))
                    if request=='2':
                        request_file = input('Attempting to find existing categories on Ecotaxa using local file data/ ecotaxa_initial_classification_template.txt as a template.\nPlease edit as needed with your categories of interest, re-save the file as ecotaxa_initial_classification.txt (very important) and press enter when ready.\nCheck existing categories at: {}'.format('https://ecotaxoserver.obs-vlfr.fr/browsetaxo/'))
                        categories=pd.read_table(path_to_git / 'data' / 'ecotaxa_initial_classification.txt',engine='python',encoding='utf-8')
                        initial_classification_list=dict(map(lambda taxon: (api_instance_taxo.search_taxa(query=taxon)[0].text,api_instance_taxo.search_taxa(query=taxon)[0].id),categories.category))
                        initial_classification_list=list(initial_classification_list.values())
                    elif request=='1':
                        initial_classification_list =[]

            elif 'ecotaxa_initial_classification_id' in cfg_metadata.keys():
                initial_classification_list=cfg_metadata['ecotaxa_initial_classification_id']

            else:
                print('Creating an Ecotaxa project empty taxonomy selection\nYou may add categories of interest through the app')
                initial_classification_list = []


            # Step 3 : Update the project with the remaining infos
            project_model = ProjectModel(projid=project_id,title=title,instrument=instrument, managers=api_response_user,init_classif_list=initial_classification_list,
                                         obj_free_cols={}, sample_free_cols={}, acquisition_free_cols={},
                                         process_free_cols={}, annotators=[], viewers=[],
                                         bodc_variables={'individual_volume': None, 'subsample_coef': None,'total_water_volume': None},
                                         contact=api_response_user[0], highest_right='Manage', license='CC BY 4.0',
                                         status = 'Annotate', objcount = None, pctvalidated = None, pctclassified = None, classifsettings = None, classiffieldlist = sorting_columns, popoverfieldlist = None,
                                         comments = '',description=description,visible=True, rf_models_used=None, cnn_network_id='')

            api_response_update = api_instance.update_project(project_id=project_id, project_model=project_model)
            update_message='Configuration file has been updated with project ID, please push to Github' if update_configuration else ''
            print('Project successfully created: https://ecotaxa.obs-vlfr.fr/gui/prj/{}.\n{}'.format(project_id,update_message))
        except Exception as e:
            project_id=None
            print("Exception when calling ProjectsApi->create_project: %s\n" % e)


    return project_id # ID of the newly created project


def upload_thumbnails_ecotaxa_project(ecotaxa_configuration,project_id,source_path):
    """
      Objective: This function uploads zip datafiles on Ecotaxa (https://ecotaxa.obs-vlfr.fr)
      Documentation available at : https://github.com/ecotaxa/ecotaxa_py_client/blob/main/docs/ProjectsApi.md#simple_import, https://github.com/ecotaxa/ecotaxa_py_client/blob/main/docs/ImportReq.md
      :param configuration: An API instance specifying the host (Ecotaxa url) and the token generated based on Ecotaxa credentials
      :param project_id: The ID of the existing Ecotaxa project. Attention: Should not already contain images.
      :param source_path: The directory containing all the datafiles generated by the **script_instrument.py**
      :return: Returns project ID
      """

    # Step 1 : Search for zip files in the source_path directory
    datafiles=[Path(source_path).expanduser()] if Path(source_path).expanduser().exists() else list(Path(source_path).expanduser().glob('*.zip'))
    if len(datafiles):
        with ecotaxa_py_client.ApiClient(ecotaxa_configuration) as api_client:
            # Create the necessary file (zip files should be temporarily stored on Ecotaxa ftp server), job (task), project (update) instances of the API
            api_instance_files = ecotaxa_py_client.FilesApi(api_client)
            api_instance_jobs = ecotaxa_py_client.JobsApi(api_client)
            api_instance = ecotaxa_py_client.ProjectsApi(api_client)
            with tqdm( desc='Working on project upload (https://ecotaxa.obs-vlfr.fr/prj/{})'.format(str(project_id)),total=len(datafiles), bar_format='{desc}{bar}', position=0, leave=True) as bar:
                for file in datafiles:

                    try:
                        # Step 2:Upload zip file to create import task request
                        api_response_file = api_instance_files.post_user_file(file=str(file),  tag='datafiles_project_{}'.format(project_id))
                        import_project_req = ImportReq(source_path=api_response_file, taxo_mappings=None,skip_loaded_files=False, skip_existing_objects=True)
                        # Step 3: Creating a task (job) to upload the zip file adn append the data to the existing project
                        api_response = api_instance.import_file(project_id, import_req=import_project_req)
                        job_id = api_response.job_id

                        # Step 4: Check the job status
                        # Insert a progress bar to allow for the job to be done based on get_job status.
                        # Attention, the break cannot be timed with job progress=(percentage) due to a small temporal offset between job progress and status
                        job_status = 'R'  # percent = 0
                        while job_status not in ('F', 'E'):  # percent!=100:
                            time.sleep(2)  # Check the job status every 2 seconds. Modify as needed
                            thread = api_instance_jobs.get_job(job_id)
                            result = thread
                            job_status = result.state
                            # Stop when job status is finished
                            if job_status == 'F':
                                break


                    except Exception as e:
                        project_id = None
                        print("Error updating project {} using {}. Please check your connection or EcoTaxa server".format(str(project_id)+' (https://ecotaxa.obs-vlfr.fr/prj/{})'.format(str(project_id)),file.stem))
                    # Step 5: Update the progress bar and move to new zip file
                    percent = np.round(100 * (bar.n / len(datafiles)), 1)
                    bar.set_description("Working on project upload {} (%s%%)".format(file.stem) % percent, refresh=True)
                    # and update progress bar
                    progress = bar.update(n=1)
        return project_id  # ID of the newly updated project

def update_thumbnails_ecotaxa_project(ecotaxa_configuration,project_id,source_path):
    """
      Objective: This function uploads zip datafiles on Ecotaxa (https://ecotaxa.obs-vlfr.fr)
      Documentation available at : https://github.com/ecotaxa/ecotaxa_py_client/blob/main/docs/ProjectsApi.md#simple_import, https://github.com/ecotaxa/ecotaxa_py_client/blob/main/docs/ImportReq.md
      :param configuration: An API instance specifying the host (Ecotaxa url) and the token generated based on Ecotaxa credentials
      :param project_id: The ID of the existing Ecotaxa project. Attention: Should not already contain images.
      :param source_path: The directory containing all the datafiles generated by the **script_instrument.py**
      :return: Returns project ID
      """

    # Step 1 : Search for zip files in the source_path directory
    datafiles=[Path(source_path).expanduser()] if Path(source_path).expanduser().exists() else list(Path(source_path).expanduser().glob('*.zip'))
    if len(datafiles):
        with ecotaxa_py_client.ApiClient(ecotaxa_configuration) as api_client:
            # Create the necessary file (zip files should be temporarily stored on Ecotaxa ftp server), job (task), project (update) instances of the API
            api_instance_files = ecotaxa_py_client.FilesApi(api_client)
            api_instance_jobs = ecotaxa_py_client.JobsApi(api_client)
            api_instance = ecotaxa_py_client.ProjectsApi(api_client)
            with tqdm( desc='Working on project update (https://ecotaxa.obs-vlfr.fr/prj/{})'.format(str(project_id)),total=len(datafiles), bar_format='{desc}{bar}', position=0, leave=True) as bar:
                for file in datafiles:

                    try:
                        # Step 2: Retrieve existing taxonomic assignments for that specific acquisition
                        api_instance_objects = objects_api.ObjectsApi(api_client)
                        api_instance_object = ecotaxa_py_client.ObjectApi(api_client)
                        api_response_samples = ecotaxa_py_client.SamplesApi(api_client).samples_search(project_ids=str(project_id), id_pattern=file.stem)
                        api_response_objects = api_instance_objects.get_object_set(project_id=int(project_id),project_filters=ProjectFilters(statusfilter="",samples=str(api_response_samples[0].sampleid)),fields="obj.orig_id,txo.display_name,obj.classif_id,obj.classif_qual,obj.classif_who,obj.classif_when,obj.classif_auto_id,obj.classif_auto_score,obj.classif_auto_when,obj.classif_crossvalidation_id")

                        # Step 3: Append the labels to ecotaxa data table and save the classification history locally
                        df_objects = pd.DataFrame(api_response_objects.details,columns=['object_origid','object_annotation_category', 'object_annotation_category_id', 'object_annotation_status','object_annotation_person_id','object_annotation_date','object_prediction_id','object_prediction_score','object_prediction_date','object_prediction_cross_validation_id']).assign(object_id=api_response_objects.to_dict().get('object_ids'))
                        api_response_users=pd.DataFrame(map(lambda user: ecotaxa_py_client.UsersApi(api_client).get_user(user_id=int(user)).to_dict(),df_objects.dropna(subset=['object_annotation_person_id']).object_annotation_person_id.unique())) if len(df_objects.dropna(subset=['object_annotation_person_id'])) else pd.DataFrame({'id':pd.NA},index=[0])
                        df_objects=pd.merge(df_objects,api_response_users.rename(columns={'name':'object_annotation_person_name','email':'object_annotation_person_email'}),how='left',left_on='object_annotation_person_id',right_on='id')
                        df_objects=df_objects.assign(object_annotation_status=lambda x: x.object_annotation_status.map({'V':'validated','P':'predicted','D':'dubious'}),object_annotation_time=lambda x: pd.to_datetime(x.object_annotation_date.astype(str),format='%Y-%m-%dT%H:%M:%S.%f',errors='ignore') if pd.to_datetime(df_objects.object_annotation_date.astype(str),format='%Y-%m-%dT%H:%M:%S.%f',errors='ignore').dtype!='O' else pd.NaT)
                        df_objects['object_annotation_time']=df_objects.object_annotation_time.dt.strftime('%H%M%S')
                        df_objects=df_objects.assign(object_annotation_date=lambda x: pd.to_datetime(x.object_annotation_date,format='%Y-%m-%dT%H:%M:%S.%f',errors='ignore') if pd.to_datetime(df_objects.object_annotation_date.astype(str),format='%Y-%m-%dT%H:%M:%S.%f',errors='ignore').dtype!='O' else pd.NaT)
                        df_objects['object_annotation_date']=df_objects.object_annotation_date.dt.strftime('%Y%m%d')
                        df_history =  pd.DataFrame(map(lambda history: history.to_dict(),list(itertools.chain(*list(map(lambda ID: api_instance_object.object_query_history(object_id=int(ID)),df_objects.object_id.unique()))))))
                        path_to_file=file.parent /file.stem / 'ecotaxa_table_{}.tsv'.format(file.stem.rstrip().replace(' ','_'))
                        df_history.to_csv(str(path_to_file).replace('ecotaxa_table_','annotations_table_'), sep='\t', index=False)
                        df_ecotaxa=pd.read_table(path_to_file,sep='\t',skiprows=1,header=None,dtype=dict_ecotaxa_types)
                        df_ecotaxa.columns=pd.read_table(path_to_file,sep='\t',nrows=1,header=None).loc[0]
                        df_ecotaxa.loc[1:,'object_time']=df_ecotaxa.loc[1:,'object_time'].astype(str).str.zfill(6)
                        df_ecotaxa.loc[1:,'object_date'] = df_ecotaxa.loc[1:,'object_date'].astype(str).str.zfill(6)
                        df_ecotaxa=pd.merge(df_ecotaxa.astype({'object_id':str})[['object_id']+[column for column in df_ecotaxa.columns if column not in df_objects.columns]],df_objects.astype({'object_origid':str})[[column for column in ['object_origid', 'object_annotation_category', 'object_annotation_status', 'object_annotation_date', 'object_annotation_time', 'object_annotation_person_name', 'object_annotation_person_email', 'object_prediction_id', 'object_prediction_score','object_prediction_date', 'object_prediction_cross_validation_id'] if column in df_objects.columns]],how='left',left_on='object_id',right_on='object_origid').drop(columns='object_origid')
                        df_ecotaxa.loc[0,[ 'object_annotation_category', 'object_annotation_status', 'object_annotation_date', 'object_annotation_time', 'object_annotation_person_name', 'object_annotation_person_email', 'object_prediction_id', 'object_prediction_score','object_prediction_date', 'object_prediction_cross_validation_id']]=['[t]','[t]','[t]','[t]','[t]','[t]','[t]','[f]','[t]','[t]']
                        df_ecotaxa_update=df_ecotaxa.copy()[['img_file_name']+[column for column in df_ecotaxa.columns if 'object_' in column]+[column for column in df_ecotaxa.columns if 'sample_' in column]+[column for column in df_ecotaxa.columns if 'acq_' in column]+[column for column in df_ecotaxa.columns if 'process_' in column]].drop(columns=['object_prediction_id', 'object_prediction_score','object_prediction_date', 'object_prediction_cross_validation_id'])
                        df_ecotaxa_update.to_csv(path_to_file, sep='\t', index=False)
                        file.unlink(missing_ok=True)
                        shutil.make_archive(str(path_to_file.parent), 'zip', path_to_file.parent, base_dir=None)

                        # Step 4: Erase objects, otherwise the thumbnails cannot be updated through regular import method
                        confirmation=input('Updating thumbnails, data, and metadata for acquisition {}. Would you like to quit (press enter) or continue (type continue)'.format(file.stem))
                        if confirmation=='':
                            continue
                        api_response_erase_objects=api_instance_objects.erase_object_set(request_body=list(map(int,df_objects.object_id.astype(int).unique())))

                        # Step 5: Upload zip file to create import task request
                        api_response_file = api_instance_files.post_user_file(file=str(file),  tag='datafiles_project_{}'.format(project_id))

                        # Step 6: Creating a task (job) to upload the zip file with pre-existing labels
                        import_project_req = ImportReq(source_path=api_response_file, taxo_mappings=None,skip_loaded_files=False, skip_existing_objects=True)
                        api_response = api_instance.import_file(project_id, import_req=import_project_req)
                        job_id = api_response.job_id

                        # Step 7: Check the job status
                        # Insert a progress bar to allow for the job to be done based on get_job status.
                        # Attention, the break cannot be timed with job progress=(percentage) due to a small temporal offset between job progress and status
                        job_status = 'R'  # percent = 0
                        while job_status not in ('F', 'E'):  # percent!=100:
                            time.sleep(2)  # Check the job status every 2 seconds. Modify as needed
                            thread = api_instance_jobs.get_job(job_id)
                            result = thread
                            job_status = result.state
                            # Stop when job status is finished
                            if job_status == 'F':
                                break
                        # Step 8: Update sample infos
                        for sample in api_response_samples:
                            bulk_update_req = ecotaxa_py_client.BulkUpdateReq(target_ids=[sample.sampleid],updates=[{'ucol':key,'uval':value}  for key,value in df_ecotaxa_update.query('sample_id=="{}"'.format(sample.orig_id)).reset_index(drop=True).loc[0,[column for column in df_ecotaxa_update.columns if 'sample' in column]].to_dict().items()])
                            api_instance_samples=ecotaxa_py_client.SamplesApi(api_client)
                            api_response = api_instance_samples.update_samples(bulk_update_req)

                    except Exception as e:
                        project_id = None
                        print("Error updating project {} using {}. Please check your connection or EcoTaxa server".format(str(project_id)+' (https://ecotaxa.obs-vlfr.fr/prj/{})'.format(str(project_id)),file.stem))
                    # Step 8: Update the progress bar and move to new zip file
                    percent = np.round(100 * (bar.n / len(datafiles)), 1)
                    bar.set_description("Working on project update {} (%s%%)".format(file.stem) % percent, refresh=True)
                    # and update progress bar
                    progress = bar.update(n=1)
        return project_id  # ID of the newly updated project

def export_ecotaxa_project_table(ecotaxa_configuration=configuration,project_id=int(cfg_metadata['ecotaxa_lexplore_alga_flowcam_micro_projectid']),export_path=Path(cfg_metadata['flowcam_10x_context_file'].replace('acquisitions', 'ecotaxa')).parent):
    """
      Objective: This function uses default option to export datatables on Ecotaxa (https://ecotaxa.obs-vlfr.fr)
      :param configuration: An API instance specifying the host (Ecotaxa url) and the token generated based on Ecotaxa credentials
      :param project_id: The ID of the existing Ecotaxa project.
      :param export_path: The storage directory
      """

    BASE_URL = "https://ecotaxa.obs-vlfr.fr"
    # Step 1 : Access project according to the info in project_id and configuration
    with ecotaxa_py_client.ApiClient(ecotaxa_configuration) as api_client:
        # Create an object instance of the API class
        api_instance = objects_api.ObjectsApi(api_client)
        body_export_object_set_object_set_export_post = BodyExportObjectSetObjectSetExportPost(
            filters=ProjectFilters(),
            # ProjectFilters(statusfilter="PVD") for Predicted, Validated, dubious. Leave empty to get unclassified ROI
            # get P(redicted), V(alidated), D(ubious) images. Check other options for filter here: https://github.com/ecotaxa/ecotaxa_py_client/blob/main/docs/ProjectFilters.md
            request=ExportReq(project_id=project_id,  # the unique project ID of interest (integer)
                              exp_type="TSV",
                              use_latin1=False,
                              tsv_entities="OPASH",
                              # entities to be exported: O(bjects), P(rocess), A(cquisition), S(ample), classification H(istory)
                              split_by="",  # no split=single table with all images/objects
                              coma_as_separator=False,  # set decimal separator to point
                              format_dates_times=False,
                              with_images=False,  # exporting images
                              with_internal_ids=False,
                              only_first_image=False,
                              sum_subtotal="A",
                              out_to_ftp=False)
        )
    # Step 2: Generate a job/task (=Export Object Set)
    try:
        api_jobresponse = api_instance.export_object_set(body_export_object_set_object_set_export_post)
    except ecotaxa_py_client.ApiException as e:
        print("Exception when calling ObjectsApi->export_object_set: %s\n" % e)

    # Report job ID. You may check the job created here: https://ecotaxa.obs-vlfr.fr/Jobs/listall
    job_id = api_jobresponse.job_id
    api_jobinstance = jobs_api.JobsApi(api_client)
    print("Creating export file with job ID: ", job_id, sep=' ')

    # Necessary break between step 3 and 4
    # Insert a progress bar to allow for the job to be done based on get_job status.
    # Attention, the break cannot be timed with job progress=(percentage) due to a small temporal offset between job progress and status
    with tqdm(desc='Working on project {} export'.format(str(project_id)), total=1000, bar_format='{desc}{bar}', position=0, leave=True) as bar:
        job_status = 'R'  # percent = 0
        while job_status not in ('F', 'E'):  # percent!=100:
            time.sleep(2)  # Check the job status every 2 seconds. Modify as needed
            thread = api_jobinstance.get_job(job_id)

            job_status = thread.state
            percent = thread.progress_pct
            bar.set_description("Working on project {} export %s%%".format(str(project_id)) % percent, refresh=True)
            # and update progress bar
            ok = bar.update(n=10)
            if job_status == 'F':
                break
            if job_status == 'E':
                print("Error creating job. Please check your connection or EcoTaxa server")
    # Step 3: Get/Download export zipfile
    zip_file = "ecotaxa_export_{}_{}Z.zip".format(str(project_id), datetime.datetime.utcnow().strftime("%Y%m%d"))  # "ecotaxa_{}".format(str(result.result['out_file']))
    path_to_zip = Path(export_path).expanduser() / zip_file
    path_to_log = Path(export_path).expanduser() / "job_{}.log".format(str(job_id))
    print("\nExporting file ", zip_file, " to ", export_path, ", please wait", sep='')

    with requests.Session() as sess:
        url = BASE_URL + "/api/jobs/%d/file" % job_id
        rsp = sess.get(url, headers={"Authorization": "Bearer " + ecotaxa_configuration.access_token}, stream=True)
        with open(path_to_zip, "wb") as fd:
            for a_chunk in rsp.iter_content():  # Loop over content, i.e. eventual HTTP chunks
                # rsp.raise_for_status()
                fd.write(a_chunk)

    print("Download completed: ", path_to_zip, "\nUnpacking zip file", sep='')
    shutil.unpack_archive(path_to_zip, export_path)  # Unzip export file
    path_to_zip.unlink(missing_ok=True)  # Delete zip file
    path_to_log.unlink(missing_ok=True)  # Delete job log

def check_ecotaxa_annotations(ecotaxa_configuration=configuration,project_id=str(cfg_metadata['ecotaxa_lexplore_alga_flowcam_micro_projectid'])):
    """
      Objective: This function checks all taxonomic categories used in set(s) of Ecotaxa project(s) (https://ecotaxa.obs-vlfr.fr)
      :param configuration: An API instance specifying the host (Ecotaxa url) and the token generated based on Ecotaxa credentials
      :param project_id: The ID of the existing Ecotaxa project. For multiple projects, use comma for separation
      :return: A dataframe with taxa ID and corresponding names
      """
    BASE_URL = "https://ecotaxa.obs-vlfr.fr"
    # Step 1 : Access project according to the info in project_id and configuration
    with ecotaxa_py_client.ApiClient(ecotaxa_configuration) as api_client:
        api_instance = ecotaxa_py_client.ProjectsApi(api_client)
        try:
            api_response = api_instance.project_set_get_stats(project_id)
            df_annotations=pd.DataFrame({'taxon':list(itertools.chain(*list(map(lambda stats: stats.used_taxa,api_response))))}).drop_duplicates(subset='taxon').reset_index(drop=True)
            api_instance = ecotaxa_py_client.TaxonomyTreeApi(api_client)
            df_annotations['EcoTaxa_hierarchy']=list(map(lambda taxon_id: api_instance.get_taxon_in_central(int(taxon_id))[0].display_name if taxon_id>0 else '',df_annotations.taxon))
        except:
            print('Error extracting project taxonomic categories. Check internet connexion and project ID')
    return df_annotations




import seaborn as sns
summary_metadata_columns=['Mode', 'Priming_Method', 'Flow_Rate', 'Recalibrations', 'Stop_Reason',
       'Sample_Volume_Aspirated', 'Sample_Volume_Processed',
       'Fluid_Volume_Imaged', 'Efficiency', 'Particle_Count', 'Total', 'Used',
       'Percentage_Used', 'Particles_Per_Image', 'Frame_Rate',
       'Background_Intensity_Mean', 'Background_Intensity_Min',
       'Background_Intensity_Max', 'Start_Time', 'Sampling_Time',
       'Environment', 'Software', 'Magnification', 'Calibration_Factor',
       'SerialNo', 'Number_of_Processors', 'Pump', 'Syringe_Size','Skip']


# Size spectrum processing
from scipy.stats import poisson
from scipy import stats
import statsmodels
from statsmodels import formula
from statsmodels.formula import api
bins=np.power(2, np.arange(-3, np.log2(100000) + 1 / 3, 1 / 3))  # Fixed size(micrometers) bins used for UVP/EcoPart data. See https://ecopart.obs-vlfr.fr/. 1/3 ESD increments allow to bin particle of doubled biovolume in consecutive bins. np.exp(np.diff(np.log((1/6)*np.pi*(EcoPart_extended_bins**3))))

df_bins = pd.DataFrame({'size_class': pd.cut(bins, bins).categories.values,  # Define bin categories (um)
                        'bin_widths': np.diff(bins),  # Define width of individual bin categories (um)
                        'range_size_bin': np.concatenate(np.diff((1 / 6) * np.pi * (np.resize( np.append(bins[0], np.append(np.repeat(bins[1:-1], repeats=2), bins[len(bins) - 1])),(len(bins) - 1, 2)) ** 3), axis=1)),  # cubic micrometers
                        'size_class_mid': stats.gmean(np.resize( np.append(bins[0], np.append(np.repeat(bins[1:-1], repeats=2), bins[len(bins) - 1])), (len(bins) - 1, 2)), axis=1)})  # Define geometrical mean of bin categories (micrometers)
def resample_biovolume(x,pixel,resample=False):
    n=np.random.poisson(lam=len(x))
    return np.abs(sum(np.random.normal(size=n,loc=np.mean(np.random.choice(x,size=n,replace=True)), scale=(1 / 6) * np.pi * (2 * ((1/(pixel*1e-06)) / np.pi) ** 0.5) ** 3))) if resample else sum(x)


def nbss_estimates(df,pixel_size,grouping_factor=['Sample'],niter=100):

    """
    Objective: This function computes the Normalized Biovolume Size Spectrum (NBSS) of individual grouping_factor levels from area estimates.\n
    The function also returns boostrapped NBSS based on the method published in Schartau et al. (2010)
    :param df: A dataframe of morphometric properties including *area* estimates for individual particles
    :param pixel_size: The camera resolution as defined by the manufacturer or calibrated with beads
    :param grouping_factor: A list including all possible variables that should be included to compute level-specific NBSS (e.g. sampling depth, taxonomic group)
    :param n_iter: The number of resampling iterations used to assess NBSS uncertainties. Default is 100 iterations
    :return: Returns a dataframe with NBSS and NBSS standard deviations based on the boostrapped NBSS and a dataframe with all boostrapped NBSS
    """

    # Assign a group index before groupby computation
    group = grouping_factor
    df_subset = pd.merge(df, df.drop_duplicates(subset=group, ignore_index=True)[group].reset_index().rename( {'index': 'Group_index'}, axis='columns'), how='left', on=group)
    # Assign size bins according to biovolume and equivalent circular diameter estimates
    df_subset=df_subset.assign(Pixel=1e+03/pixel_size,Area=lambda x:(x.area*(pixel_size**2)),Biovolume=lambda x:(1/6)*np.pi*(2*((x.area*(pixel_size**2))/np.pi)**0.5)**3,ECD=lambda x:2*((x.area*(pixel_size**2))/np.pi)**0.5)
    df_subset=df_subset.assign(size_class=lambda x:pd.cut(x.ECD,bins,include_lowest=True).astype(str))
    df_subset=pd.merge(df_subset,df_bins.astype({'size_class':str}),how='left',on=['size_class'])
    df_subset=df_subset.assign(size_class_pixel=lambda z:z.size_class_mid/pixel_size,ROI_number=1)

    # Pivot table after summing all biovolume for each size class
    x_pivot=df_subset.pivot_table(values='Biovolume',columns=['size_class_mid','range_size_bin','size_class'],index=grouping_factor+['Group_index','volume'],aggfunc=lambda z:resample_biovolume(z,pixel=1e+03/pixel_size,resample=False))
    # Normalize to size class width
    x_pivot=x_pivot/x_pivot.columns.levels[1]
    # Normalize to volume imaged
    size_class=x_pivot.columns.levels[2]
    size_class =size_class.drop('nan') if 'nan' in size_class else size_class
    x_pivot=(x_pivot.fillna(0)/pd.DataFrame(np.resize(np.repeat(x_pivot.index.get_level_values('volume'),x_pivot.shape[1]),(x_pivot.shape[0],x_pivot.shape[1])),index=x_pivot.index,columns=x_pivot.columns)).reset_index(col_level=2).droplevel(level=[0,1],axis=1)
    # Append number of observations and pixel size
    x_nbr=df_subset.pivot_table(values='ROI_number',columns=['size_class'],index=grouping_factor+['Group_index','volume'],aggfunc='sum')
    x_nbr=(x_nbr.fillna(0).cumsum(axis=0)).reset_index()
    # Melt datatable and sort by grouping factors and size classes
    x_melt=pd.merge((x_pivot.melt(id_vars=grouping_factor+['Group_index','volume'],value_vars=size_class,var_name='size_class',value_name='NBSS').replace({'NBSS':0}, np.nan)).astype({'size_class':str}),df_bins.astype({'size_class':str}),how='left',on='size_class').sort_values(grouping_factor+['Group_index','volume']+['size_class_mid']).reset_index(drop=True)
    x_melt=pd.merge(x_melt,x_nbr.melt(id_vars=grouping_factor+['Group_index','volume'],value_vars=size_class,var_name='size_class',value_name='NBSS_count').replace({'NBSS_count':0}, np.nan).astype({'size_class':str}),how='left',on=grouping_factor+['Group_index','volume','size_class'])
    x_melt=pd.merge(x_melt,((2*(df_subset.astype({'size_class':str}).groupby(['size_class']).Area.mean()/np.pi)**0.5/(df_subset.Pixel.unique()[0]*1e-03)).reset_index().rename(columns={'Area':'size_class_pixel'})),how='left',on='size_class')
    # Append boostrapped summaries
    df_nbss_std=list(map(lambda iter:(x_pivot:=df_subset.pivot_table(values='Biovolume',columns=['size_class_mid','range_size_bin','size_class'],index=grouping_factor+['Group_index','volume'],aggfunc=lambda z:resample_biovolume(z,pixel=df_subset.Pixel.unique()[0],resample=True)),x_pivot:=x_pivot/x_pivot.columns.levels[1],   x_pivot:=(x_pivot.fillna(0)/pd.DataFrame(np.resize(np.repeat(x_pivot.index.get_level_values('volume'),x_pivot.shape[1]),(x_pivot.shape[0],x_pivot.shape[1])),index=x_pivot.index,columns=x_pivot.columns)).reset_index(col_level=2).droplevel(level=[0,1],axis=1).melt(id_vars=grouping_factor+['Group_index','volume'],value_vars=size_class,var_name='size_class',value_name='NBSS').replace({'NBSS':0}, np.nan).sort_values(grouping_factor+['Group_index','volume']).reset_index(drop=True))[-1],np.arange(niter)))
    df_nbss_boot=reduce(lambda left, right: pd.merge(left, right, on=grouping_factor+['Group_index','volume','size_class'],suffixes=(None,'_right')),df_nbss_std).reset_index(drop=True).sort_values(grouping_factor+['Group_index']).rename(columns={'NBSS_right':'NBSS'})
    df_summary=pd.concat([df_nbss_boot.reset_index().drop(columns='NBSS'),pd.DataFrame({'NBSS_std':df_nbss_boot[['NBSS']].std(axis=1)})],axis=1)
    df_nbss_boot = pd.merge(df_nbss_boot, df_bins.astype({'size_class': str}), how='left', on=['size_class'])
    x_melt = pd.merge(x_melt,df_summary , how='left', on=grouping_factor+['Group_index','volume','size_class'])

    return x_melt,df_nbss_boot

def as_float_cytosense_depth(str):
    """
    Objective: This function is used to retrieve the sampling depth from the name of CytoSense datafiles (no other export options). Default is 0 unless the Depth Profile Sampler was activated
    """

    try:
        return float(str)
    except:
        return 0

def format_id_to_skip(list_id):
    """
    Objective: This function format a list of consecutive or discontinuous sequences of numbers into a string to match the expected format of particle IDs to skip
    """
    s = e = None
    for i in sorted(list_id):
        if s is None:
            s = e = i
        elif i == e or i == e + 1:
            e = i
        else:
            yield (s, e)
            s = e = i
    if s is not None:
        yield (s, e)

# Define a dictionary to specify the columns types returned by the region of interest properties table
dict_cytosense_listmode={'Particle ID':'Particle_ID','FWS Length':'fws_length','FWS Total':'fws_total','FWS Maximum':'fws_max','SWS Length':'sws_length','SWS Total':'sws_total','SWS Maximum':'sws_max','FL Yellow Length':'fly_length','FL Yellow Total':'fly_total','FL Yellow Maximum':'fly_max','FL Orange Length':'flo_length','FL Orange Total':'flo_total','FL Orange Maximum':'flo_max','FL Green Length':'flg_length','FL Green Total':'flg_total','FL Green Maximum':'flg_max','FL Red Length':'flr_length','FL Red Total':'flr_total','FL Red Maximum':'flr_max'}
dict_properties_types={'nb_particles':['[f]'], 'area':['[f]'], 'area_bbox':['[f]'],
       'area_convex':['[f]'], 'area_filled':['[f]'], 'axis_major_length':['[f]'], 'axis_minor_length':['[f]'],
       'bbox-0':['[f]'], 'bbox-1':['[f]'], 'bbox-2':['[f]'], 'bbox-3':['[f]'], 'centroid_local-0':['[f]'],
       'centroid_local-1':['[f]'], 'centroid_weighted_local-0':['[f]'],
       'centroid_weighted_local-1':['[f]'], 'eccentricity':['[f]'], 'equivalent_diameter_area':['[f]'],
       'extent':['[f]'], 'image_intensity':['[t]'], 'inertia_tensor-0-0':['[f]'], 'inertia_tensor-0-1':['[f]'],
       'inertia_tensor-1-0':['[f]'], 'inertia_tensor-1-1':['[f]'], 'inertia_tensor_eigvals-0':['[f]'],
       'inertia_tensor_eigvals-1':['[f]'], 'intensity_mean':['[f]'], 'intensity_max':['[f]'],
       'intensity_min':['[f]'], 'intensity_std':['[f]'], 'moments-0-0':['[f]'], 'moments-0-1':['[f]'],
       'moments-0-2':['[f]'], 'moments-0-3':['[f]'], 'moments-1-0':['[f]'], 'moments-1-1':['[f]'],
       'moments-1-2':['[f]'], 'moments-1-3':['[f]'], 'moments-2-0':['[f]'], 'moments-2-1':['[f]'],
       'moments-2-2':['[f]'], 'moments-2-3':['[f]'], 'moments-3-0':['[f]'], 'moments-3-1':['[f]'],
       'moments-3-2':['[f]'], 'moments-3-3':['[f]'], 'moments_central-0-0':['[f]'],
       'moments_central-0-1':['[f]'], 'moments_central-0-2':['[f]'], 'moments_central-0-3':['[f]'],
       'moments_central-1-0':['[f]'], 'moments_central-1-1':['[f]'], 'moments_central-1-2':['[f]'],
       'moments_central-1-3':['[f]'], 'moments_central-2-0':['[f]'], 'moments_central-2-1':['[f]'],
       'moments_central-2-2':['[f]'], 'moments_central-2-3':['[f]'], 'moments_central-3-0':['[f]'],
       'moments_central-3-1':['[f]'], 'moments_central-3-2':['[f]'], 'moments_central-3-3':['[f]'],
       'moments_hu-0':['[f]'], 'moments_hu-1':['[f]'], 'moments_hu-2':['[f]'], 'moments_hu-3':['[f]'],
       'moments_hu-4':['[f]'], 'moments_hu-5':['[f]'], 'moments_hu-6':['[f]'], 'num_pixels':['[f]'],
       'orientation':['[f]'], 'perimeter':['[f]'], 'slice':['[t]'],'fws_length':['[f]'],'fws_total':['[f]'],
        'fws_max':['[f]'],'sws_length':['[f]'],'sws_total':['[f]'],'sws_max':['[f]'],'fly_length':['[f]'],'fly_total':['[f]'],'fly_max':['[f]'],
        'flg_length':['[f]'],'flg_total':['[f]'],'flg_max':['[f]'],'flo_length':['[f]'],'flo_total':['[f]'],'flo_max':['[f]'],'flr_length':['[f]'],'flr_total':['[f]'],'flr_max':['[f]']}
dict_properties_types_colors=dict_properties_types | {'centroid_weighted_local-0-0':['[f]'],'centroid_weighted_local-0-1':['[f]'],'centroid_weighted_local-0-2':['[f]'], 'centroid_weighted_local-1':['[f]'],'centroid_weighted_local-1-1':['[f]'],'centroid_weighted_local-1-2':['[f]'], 'intensity_mean':['[f]'], 'intensity_max':['[f]'], 'intensity_min':['[f]'], 'intensity_std':['[f]'], 'intensity_mean-0':['[f]'], 'intensity_max-0':['[f]'], 'intensity_min-0':['[f]'], 'intensity_std-0':['[f]'],'intensity_mean-1':['[f]'], 'intensity_max-1':['[f]'], 'intensity_min-1':['[f]'], 'intensity_std-1':['[f]'],'intensity_mean-2':['[f]'], 'intensity_max-2':['[f]'], 'intensity_min-2':['[f]'], 'intensity_std-2':['[f]']}
dict_properties_visual_spreadsheet={'Name':['[t]'], 'Visualspreadsheet Area (ABD)':['[f]'], 'Visualspreadsheet Area (Filled)':['[f]'], 'Visualspreadsheet Aspect Ratio':['[f]'], 'Visualspreadsheet Average Blue':['[f]'],
       'Visualspreadsheet Average Green':['[f]'], 'Visualspreadsheet Average Red':['[f]'], 'Visualspreadsheet Calibration Factor':['[f]'],
       'Visualspreadsheet Calibration Image':['[f]'], 'Visualspreadsheet Capture ID':['[t]'] ,'Visualspreadsheet Capture X':['[f]'], 'Visualspreadsheet Capture Y':['[f]'], 'Visualspreadsheet Circularity':['[f]'],
       'Visualspreadsheet Circularity (Hu)':['[f]'], 'Visualspreadsheet Compactness':['[f]'], 'Visualspreadsheet Convex Perimeter':['[f]'], 'Visualspreadsheet Convexity':['[f]'],
       'Visualspreadsheet Date':['[t]'], 'Visualspreadsheet Diameter (ABD)':['[f]'], 'Visualspreadsheet Diameter (ESD)':['[f]'], 'Visualspreadsheet Diameter (FD)':['[f]'],
       'Visualspreadsheet Edge Gradient':['[f]'], 'Visualspreadsheet Elapsed Time':['[f]'], 'Visualspreadsheet Elongation':['[f]'], 'Visualspreadsheet Feret Angle Max':['[f]'],
       'Visualspreadsheet Feret Angle Min':['[f]'], 'Visualspreadsheet Fiber Curl':['[f]'], 'Visualspreadsheet Fiber Straightness':['[f]'], 'Visualspreadsheet Filter Score':['[f]'],
       'Visualspreadsheet Geodesic Aspect Ratio':['[f]'], 'Visualspreadsheet Geodesic Length':['[f]'], 'Visualspreadsheet Geodesic Thickness':['[f]'],
       'Visualspreadsheet Group ID':['[t]'], 'Visualspreadsheet Image Height':['[f]'], 'Visualspreadsheet Image Width':['[f]'], 'Visualspreadsheet Image X':['[f]'], 'Visualspreadsheet Image Y':['[f]'],
       'Visualspreadsheet Length':['[f]'], 'Visualspreadsheet Particles Per Chain':['[f]'], 'Visualspreadsheet Perimeter':['[f]'], 'Visualspreadsheet Ratio Blue/Green':['[f]'],
       'Visualspreadsheet Ratio Red/Blue':['[f]'], 'Visualspreadsheet Ratio Red/Green':['[f]'], 'Visualspreadsheet Roughness':['[f]'], 'Visualspreadsheet Source Image':['[f]'],
       'Visualspreadsheet Sum Intensity':['[f]'], 'Visualspreadsheet Symmetry':['[f]'], 'Visualspreadsheet Time':['[t]'], 'Visualspreadsheet Transparency':['[f]'], 'Visualspreadsheet Volume (ABD)':['[f]'],
       'Visualspreadsheet Width':['[f]']}
#Define a metadata table for Lexplore based on the metadata confiugration file
df_sample_lexplore_cytosense=pd.DataFrame({'sample_longitude':['[f]',cfg_metadata['longitude']],'sample_latitude':['[f]',cfg_metadata['latitude']],'sample_platform':['[t]',cfg_metadata['program']],'sample_project':['[t]',cfg_metadata['project']],'sample_principal_investigator':['[t]',cfg_metadata['principal_investigator']],'sample_depthprofile':['[t]','True'] })
df_acquisition_lexplore_cytosense=pd.DataFrame({'acq_instrument':['[t]','CytoSense_CS-2015-71'],'acq_pumptype':['[t]','peristaltic'],'acq_software':['[t]',cfg_metadata['version_cytoUSB']],'acq_pixel_um':['[f]',cfg_metadata['pixel_size_cytosense']],'acq_frequency_fps':['[f]',cfg_metadata['fps_cytosense']],'acq_max_width_um':['[f]',cfg_metadata['width_cytosense']*cfg_metadata['pixel_size_cytosense']],'acq_max_height_um':['[f]',cfg_metadata['height_cytosense']*cfg_metadata['pixel_size_cytosense']]})
df_processing_lexplore_cytosense=pd.DataFrame({'process_operator':['[t]',cfg_metadata['instrument_operator']],'process_code':['[t]','https://github.com/mdugenne/Lexplore_ALGA'],'process_min_diameter_um':['[f]',10],'process_max_diameter_um':['[f]',cfg_metadata['width_cytosense']*cfg_metadata['pixel_size_cytosense']],'process_gamma':['[f]',0.7]})

df_context_flowcam_micro=pd.read_csv(r'{}'.format(cfg_metadata['flowcam_10x_context_file']),sep=r'\=|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains(r"\[",case=True)').set_index('Name').T.rename(index={'Value':Path(cfg_metadata['flowcam_10x_context_file']).stem})
df_context_flowcam_macro=pd.read_csv(r'{}'.format(cfg_metadata['flowcam_macro_context_file']),sep=r'\=|\t',engine='python',encoding='latin-1',names=['Name','Value']).query('not Name.str.contains(r"\[",case=True)').set_index('Name').T.rename(index={'Value':Path(cfg_metadata['flowcam_macro_context_file']).stem})
# Remove duplicated columns
df_context_flowcam_macro=df_context_flowcam_macro.loc[:,~df_context_flowcam_macro.columns.duplicated()].copy()
df_context_flowcam_micro=df_context_flowcam_micro.loc[:,~df_context_flowcam_micro.columns.duplicated()].copy()
# Merge in a single context file
df_context=pd.concat([pd.concat([df_context_flowcam_micro,pd.DataFrame(dict(zip([column for column in df_context_flowcam_macro.columns if column not in df_context_flowcam_micro.columns],[pd.NA]*len([column for column in df_context_flowcam_macro.columns if column not in df_context_flowcam_micro.columns]))),index=df_context_flowcam_micro.index)],axis=1),pd.concat([df_context_flowcam_macro,pd.DataFrame(dict(zip([column for column in df_context_flowcam_micro.columns if column not in df_context_flowcam_macro.columns],[pd.NA]*len([column for column in df_context_flowcam_micro.columns if column not in df_context_flowcam_macro.columns]))),index=df_context_flowcam_macro.index)],axis=1)],axis=0)


df_sample_lexplore_flowcam=pd.DataFrame({'sample_longitude':['[f]',cfg_metadata['longitude']],'sample_latitude':['[f]',cfg_metadata['latitude']],'sample_platform':['[t]',cfg_metadata['program']],'sample_project':['[t]',cfg_metadata['project']],'sample_principal_investigator':['[t]',cfg_metadata['principal_investigator']],'sample_depthprofile':['[t]','False'],'sample_storage_gear':['[t]','wasam'],'sample_fixative':['[t]','glutaraldehyde'],'sample_fixative_final_concentration':['[f]',0.25] })
df_acquisition_lexplore_flowcam_10x=pd.DataFrame({'acq_instrument':['[t]','Flowcam_SN_{}'.format(df_context_flowcam_micro.SerialNo.astype(str).values[0])],'acq_software':['[t]',df_context_flowcam_micro.SoftwareName.astype(str).values[0]+'_v'+df_context_flowcam_micro.SoftwareVersion.astype(str).values[0]],'acq_flowcell':['[t]',df_context_flowcam_micro.FlowCellType.astype(str).values[0]],'acq_objective':['[t]','{}x'.format(df_context_flowcam_micro.CameraMagnification.astype(str).values[0])],'acq_mode':['[t]','Autoimage'],'acq_stop_criterion':['[t]','Volume_pumped'],'acq_pumptype':['[t]','syringe'],'acq_max_width_um':['[f]',(df_context_flowcam_micro.AcceptableRight.astype(float).values[0]-df_context_flowcam_micro.AcceptableLeft.astype(float).values[0])*cfg_metadata['pixel_size_flowcam_10x']],'acq_max_height_um':['[f]',(df_context_flowcam_micro.AcceptableBottom.astype(float).values[0]-df_context_flowcam_micro.AcceptableTop.astype(float).values[0])*cfg_metadata['pixel_size_flowcam_10x']],'acq_min_diameter_um':['[f]',df_context_flowcam_micro.MinESD.astype(float).values[0]],'acq_max_diameter_um':['[f]',df_context_flowcam_micro.MaxESD.astype(float).values[0]],'acq_dark_threshold':['[f]',df_context_flowcam_micro.ThresholdDark.astype(float).values[0]],'acq_dark_threshold':['[f]',df_context_flowcam_micro.ThresholdDark.astype(float).values[0]],'acq_light_threshold':['[f]',df_context_flowcam_micro.ThresholdLight.astype(float).values[0]],'acq_neighbor_distance':['[f]',df_context_flowcam_micro.DistanceToNeighbor.astype(float).values[0]],'acq_closing_iterations':['[f]',df_context_flowcam_micro.CloseHoles.astype(float).values[0]],'acq_rolling_calibration':['[f]',df_context_flowcam_micro.FrameCount.astype(float).values[0]]})
df_processing_lexplore_flowcam_10x=pd.DataFrame({'process_operator':['[t]',cfg_metadata['instrument_operator']],'process_code':['[t]','https://github.com/mdugenne/Lexplore_ALGA'],'process_imagetype':['[t]','mosaic']})
df_acquisition_lexplore_flowcam_macro=pd.DataFrame({'acq_instrument':['[t]','Flowcam_SN_{}'.format(df_context_flowcam_macro.SerialNo.astype(str).values[0])],'acq_software':['[t]',df_context_flowcam_macro.SoftwareName.astype(str).values[0]+'_v'+df_context_flowcam_macro.SoftwareVersion.astype(str).values[0]],'acq_flowcell':['[t]',df_context_flowcam_macro.FlowCellType.astype(str).values[0]],'acq_objective':['[t]','{}x'.format(df_context_flowcam_macro.CameraMagnification.astype(str).values[0])],'acq_mode':['[t]','Autoimage'],'acq_stop_criterion':['[t]','Volume_pumped'],'acq_pumptype':['[t]','peristaltic'],'acq_max_width_um':['[f]',(df_context_flowcam_macro.AcceptableRight.astype(float).values[0]-df_context_flowcam_macro.AcceptableLeft.astype(float).values[0])*cfg_metadata['pixel_size_flowcam_macro']],'acq_max_height_um':['[f]',(df_context_flowcam_macro.AcceptableBottom.astype(float).values[0]-df_context_flowcam_macro.AcceptableTop.astype(float).values[0])*cfg_metadata['pixel_size_flowcam_macro']],'acq_min_diameter_um':['[f]',df_context_flowcam_macro.MinESD.astype(float).values[0]],'acq_max_diameter_um':['[f]',df_context_flowcam_macro.MaxESD.astype(float).values[0]],'acq_dark_threshold':['[f]',df_context_flowcam_macro.ThresholdDark.astype(float).values[0]],'acq_dark_threshold':['[f]',df_context_flowcam_macro.ThresholdDark.astype(float).values[0]],'acq_light_threshold':['[f]',df_context_flowcam_macro.ThresholdLight.astype(float).values[0]],'acq_neighbor_distance':['[f]',df_context_flowcam_macro.DistanceToNeighbor.astype(float).values[0]],'acq_closing_iterations':['[f]',df_context_flowcam_macro.CloseHoles.astype(float).values[0]],'acq_rolling_calibration':['[f]',df_context_flowcam_macro.FrameCount.astype(float).values[0]]})
df_processing_lexplore_flowcam_macro=pd.DataFrame({'process_operator':['[t]',cfg_metadata['instrument_operator']],'process_code':['[t]','https://github.com/mdugenne/Lexplore_ALGA'],'process_imagetype':['[t]','mosaic']})

def generate_ecotaxa_table(df,instrument,path_to_storage=None):
    """
    Objective: This function generates a table to be uploaded on Ecotaxa along with the thumbnails after acquisitions with the CytoSense or FlowCam imaging flow cytometer
    :param df: A dataframe of morphometric properties including *area* estimates for individual particles. \nFor FlowCam acquisitions, this tables corresponds to the "data" csv file that can be exported automatically (in context file, go to the Reports tab and enable option "Export list data when run terminates") or manually (in the main window, click on the File tab and select "Export data". Note that if the run was closed you'll need to re-open the data by clicking on "Open data" amd selecting the run).
    \nFor CytoSense acquisitions, the table is generated by running the custom 0_upload_thumbnails_cytosense.py. This script uses cropped images (without scalebars) exported automatically after each acquisition.
    :param instrument: Instrument used for the acquisition. Supported options are "CytoSense" or "FlowCam"
    :param path_to_storage: A Path object specifying where the table should be saved. If None, the function returns the table that is not saved.
    :return: Returns a dataframe including with the appropriate format for upload on Ecotaxa. See instructions at https://ecotaxa.obs-vlfr.fr/gui/prj/
    """
    if instrument.lower()=='cytosense':
        df_ecotaxa_object=pd.concat([pd.DataFrame(
            {'img_file_name': ['[t]'] + list(df.img_file_name.values),
             'object_id': ['[t]'] + list(df.img_file_name.astype(str)), # Attention: The object_id needs to include the file extension
             'object_lat': ['[f]'] + [cfg_metadata['latitude']] * len(df),
             'object_lon': ['[f]'] + [cfg_metadata['longitude']] * len(df),
             'object_date': ['[t]'] + [df.Sample.astype(str).values[0].split('_')[-2].replace('-','')] * len(df),
             'object_time': ['[t]'] + [df.Sample.astype(str).values[0].split('_')[-1].replace('h','')+'00'] * len(df),
             'object_depth_min': ['[f]'] + [as_float_cytosense_depth(df.Sample.astype(str).values[0][:-17].split('_')[-1])/100] * len(df),
             'object_depth_max': ['[f]'] + [as_float_cytosense_depth(df.Sample.astype(str).values[0][:-17].split('_')[-1])/100] * len(df)
             }),pd.concat([pd.DataFrame(dict(zip(('object_'+pd.Series([key for key in dict_properties_types.keys() if key in df.columns])).values,[value for key, value in dict_properties_types.items() if key in df.columns]))),df.rename(columns=dict(zip(list(df.columns),list('object_'+df.columns))))[list(('object_'+pd.Series([key for key in dict_properties_types.keys() if key in df.columns]).values))]],axis=0).drop(columns=['object_image_intensity','object_slice']).reset_index(drop=True)],axis=1).reset_index(drop=True)
        df_ecotaxa_sample=pd.concat([pd.concat([pd.DataFrame({'sample_id': ['[t]'] + list(df.Sample.astype(str).values)}),pd.concat([df_sample_lexplore_cytosense,*[df_sample_lexplore_cytosense.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True)],axis=1),pd.DataFrame({'sample_volume_analyzed_ml': ['[f]'] + list(df.sample_volume_analyzed_ml.values),'sample_volume_pumped_ml': ['[f]'] + list(df.sample_volume_pumped_ml.values),'sample_volume_fluid_imaged_ml': ['[f]'] + list(df.sample_volume_fluid_imaged_ml.values),'sample_duration_sec': ['[f]'] + list(df.sample_duration_sec.values)})],axis=1)
        df_ecotaxa_acquisition=pd.concat([pd.concat([df_acquisition_lexplore_cytosense,*[df_acquisition_lexplore_cytosense.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True),pd.DataFrame({'acq_flow_rate': ['[f]'] + list(df.sample_flow_rate.astype(float).values),'acq_trigger_channel': ['[t]'] + list(df.sample_trigger.astype(str).values),'acq_trigger_threshold_mv': ['[f]'] + list(df.sample_trigger_threshold_mv.astype(float).values)})],axis=1).assign(acq_id=['[t]'] +list(df.Sample.astype(str).values))
        df_ecotaxa_process=pd.concat([df_processing_lexplore_cytosense,*[df_processing_lexplore_cytosense.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True).assign(process_id=['[t]'] +['process_Lexplore_ALGA']*len(df))
        df_ecotaxa=pd.concat([df_ecotaxa_object,df_ecotaxa_sample,df_ecotaxa_acquisition[['acq_id']+list(df_ecotaxa_acquisition.columns[0:-1])],df_ecotaxa_process[['process_id']+list(df_ecotaxa_process.columns[0:-1])]],axis=1)

    elif instrument.lower()=='flowcam':
        df_ecotaxa_object=pd.concat([pd.DataFrame(
            {'img_file_name': ['[t]'] + list(df.img_file_name.values),
             'object_id': ['[t]'] + list(df.img_file_name.astype(str)), # Attention: The object_id needs to include the file extension
             'object_lat': ['[f]'] + [cfg_metadata['latitude']] * len(df),
             'object_lon': ['[f]'] + [cfg_metadata['longitude']] * len(df),
             'object_date': ['[t]'] + [df.Sample.astype(str).values[0].split('_')[5].replace('-','')] * len(df),
             'object_time': ['[t]'] + [cfg_metadata['wasam_sampling_time']] * len(df),
             'object_depth_min': ['[f]'] + [cfg_metadata['wasam_sampling_depth']] * len(df),
             'object_depth_max': ['[f]'] + [cfg_metadata['wasam_sampling_depth']] * len(df)
             }),pd.concat([pd.DataFrame(dict(zip(('object_'+pd.Series([key.replace(' ','_').replace('/','_').replace('(','').replace(')','').lower() for key in {**dict_properties_types_colors,**dict_properties_visual_spreadsheet}.keys() if key in df.columns])).values,[value for key,value in {**dict_properties_types_colors,**dict_properties_visual_spreadsheet}.items() if key in df.columns]))),df.rename(columns=dict(zip([column for column in df.columns if column in {**dict_properties_types_colors,**dict_properties_visual_spreadsheet}.keys()],['object_'+column.replace(' ','_').replace('/','_').replace('(','').replace(')','').lower() for column in df.columns if column in {**dict_properties_types_colors,**dict_properties_visual_spreadsheet}.keys()])))[['object_'+column.replace(' ','_').replace('/','_').replace('(','').replace(')','').lower() for column in df.columns if column in {**dict_properties_types_colors,**dict_properties_visual_spreadsheet}.keys()]]],axis=0).drop(columns=['object_image_intensity','object_slice']).reset_index(drop=True)],axis=1).reset_index(drop=True)

        df_ecotaxa_sample=pd.concat([pd.concat([pd.DataFrame({'sample_id': ['[t]'] + list(df.Sample.astype(str).values)}),pd.concat([df_sample_lexplore_flowcam,*[df_sample_lexplore_flowcam.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True)],axis=1),pd.DataFrame({'sample_volume_analyzed_ml': ['[f]'] + list(df.sample_volume_analyzed_ml.values),'sample_volume_pumped_ml': ['[f]'] + list(df.sample_volume_pumped_ml.values),'sample_volume_fluid_imaged_ml': ['[f]'] + list(df.sample_volume_fluid_imaged_ml.values),'sample_duration_sec': ['[t]'] + list(df.sample_duration_sec.values)})],axis=1)
        df_acq=df_acquisition_lexplore_flowcam_10x if any(df.Sample.str.lower().str.split('_').str[0:2].str.join('_')=='flowcam_10x') else df_acquisition_lexplore_flowcam_macro if any(df.Sample.str.lower().str.split('_').str[0:2].str.join('_')=='flowcam_2mm')  else pd.DataFrame()
        df_ecotaxa_acquisition=pd.concat([pd.concat([df_acq,*[df_acq.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True),pd.DataFrame({'acq_flow_rate': ['[f]'] + list(df.sample_flow_rate.astype(float).values),'acq_acquisition_date':['[t]']+list(df.acq_acquisition_date.astype(str).values)})],axis=1).assign(acq_id=['[t]'] +list(df.Sample.astype(str).values))
        df_process = df_processing_lexplore_flowcam_10x if any(df.Sample.str.lower().str.split('_').str[0:2].str.join('_') == 'flowcam_10x') else df_processing_lexplore_flowcam_macro if any(df.Sample.str.lower().str.split('_').str[0:2].str.join('_') == 'flowcam_2mm') else pd.DataFrame()
        df_ecotaxa_process=pd.concat([df_process,*[df_process.tail(1)]*(len(df)-1)],axis=0).reset_index(drop=True).assign(process_id=['[t]'] +['process_Lexplore_ALGA']*len(df))
        df_ecotaxa=pd.concat([df_ecotaxa_object,df_ecotaxa_sample,df_ecotaxa_acquisition[['acq_id']+list(df_ecotaxa_acquisition.columns[0:-1])],df_ecotaxa_process[['process_id']+list(df_ecotaxa_process.columns[0:-1])]],axis=1)

    else:
        print('Instrument not supported, please specify either "CytoSense" or "FlowCam". Quitting')
    if len(path_to_storage):
        df_ecotaxa.to_csv(path_to_storage, sep='\t',index=False)
    return df_ecotaxa

def season(date,hemisphere):
    """
    Objective: This function returns the season of a sampling time
    :param date: A datetime object
    :param hemisphere: "north" or "south"
    :return: Returns the assigned season
    """
    md = date.month * 100 + date.day

    if ((md > 320) and (md < 621)):
        s = 0 #spring
    elif ((md > 620) and (md < 923)):
        s = 1 #summer
    elif ((md > 922) and (md < 1223)):
        s = 2 #fall
    else:
        s = 3 #winter

    if hemisphere != 'north':
        if s < 2:
            s += 2
        else:
            s -= 2

    return {0:'Spring',1:'Summer',2:'Fall',3:'Winter'}[s]

# Function to standardize taxa list using the World Register of Marine (and freshwater) Species
def annotation_in_WORMS(hierarchy):
    """
    Objective: this function uses python requests to search taxonomic annotation on the World Register of Marine Sepcies (WORMS, https://www.marinespecies.org/).
    Note: Only accepted names will be used.\n
    :param hierarchy: Single or hierarchical taxonomic annotation to standardize with WORMS. If hierarchical taxonomic annotation, use > to separate taxonomic ranks
    :return: dataframe with corresponding rank, domain, phylum, class, order, family, genus, functional group, Taxon url, reference, citation, and URL for the annotation
    """
    hierarchy_levels = re.split('[<>]',hierarchy)
    df_hierarchy = pd.DataFrame({'EcoTaxa_hierarchy': hierarchy, 'Full_hierarchy': '', 'Rank': '', 'Type': '', 'Domain': '', 'Phylum': '','Class': '', 'Order': '', 'Family': '', 'Genus': '', 'Functional_group': '', 'WORMS_ID': '', 'Reference': '','Citation': ''}, index=[0])
    url = 'https://www.marinespecies.org/aphia.php?p=taxlist'

    for annotation in hierarchy_levels[::-1]:
        with HTMLSession() as session:
            data={'searchpar':'0','tName':annotation,'marine':'0','fossil':'4'}
            # Turn marine only and extant only search off
            #session.post('https://www.marinespecies.org/aphia.php?p=search',data=data)
            taxon_list=session.post(url=url,data=data, headers={'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'})
            soup = BeautifulSoup(taxon_list.content, "lxml")
            if len(soup.find_all(class_="list-group-item")) == 0:  # If post results in a single page:
                if len(soup.find_all(class_="alert alert-info")) > 0:
                    continue  # Skip current level if annotation not found
                else:
                    taxo_soup=soup
                    attributes_soup=''
                    script=''
                    annotation_webpage_with_attributes = session.get(urljoin(taxon_list.url, '#attributes'))
                    if annotation_webpage_with_attributes.ok:#len(list(filter(None,[id.get('id') if id.get('id') == "aphia_attributes_group_show" else id.get('href') if id.get('href') == '#attributes' else None for id in soup.findAll('a')]))) > 0:
                        attributes_soup = BeautifulSoup(annotation_webpage_with_attributes.content, 'lxml')
                        script = attributes_soup.find_all('script')
                        if len(attributes_soup.findAll(attrs={'id': 'aphia_attributes_group'})):
                            if 'No attributes found on lower taxonomic level' in attributes_soup.findAll(attrs={'id': 'aphia_attributes_group'})[0].getText() or len([item.getText() for item in script if 'Functional group' in item.getText()])==0:
                                attributes_soup = ''
                                script = ''
            else: # If post result in a list of taxa
                search_response = pd.DataFrame({'Taxon': [item.getText() for item in soup.find_all(class_="list-group-item") if any(re.findall(r'unaccepted|uncertain|unassessed',item.getText())) == False],'Link': [item.find('a').get('href') for item in soup.find_all(class_="list-group-item") if any(re.findall(r'unaccepted|uncertain|unassessed',item.getText())) == False]})
                if len(search_response) == 0:
                    continue  # Skip current level if annotation not found
                if search_response.Taxon[0].split(' ')[0].lower() not in  annotation.lower():
                    continue  # Skip current level if search response different from level
                else:

                    status='wait'
                    while status!='ok':
                        annotation_webpage = session.get(urljoin(url, search_response.Link[0]),timeout=5,stream=True)
                        status='ok' if annotation_webpage.status_code==200 else 'wait'
                    time.sleep(5)
                    taxo_soup = BeautifulSoup(annotation_webpage.content, 'lxml')
                    if len(taxo_soup.find_all(class_="alert alert-info")) > 0:
                        while status != 'ok':
                            res = session.post(urljoin(url, search_response.Link[0]),timeout=5,stream=True)
                            status = 'ok' if res.status_code == 200 else 'wait'

                        time.sleep(5)
                        soup = BeautifulSoup(res.content, "lxml")
                        taxo_soup = BeautifulSoup(res.content, 'lxml')
                        attributes_soup = ''
                        script = ''
                        annotation_webpage_with_attributes = session.get(urljoin(annotation_webpage.url,'#attributes'))  # get_attributes_output(url='https://www.marinespecies.org/aphia.php?p=search&adv=1',id=annotation_webpage.url.split('id=')[-1])
                        if annotation_webpage_with_attributes.ok:#len(list(filter(None,[id.get('id') if id.get('id') == "aphia_attributes_group_show" else id.get('href') if id.get('href') == '#attributes' else None for id in taxo_soup.findAll('a')]))) > 0:  # taxo_soup.findAll('a',onclick=True)[0].get('id')=="aphia_attributes_group_show":
                            attributes_soup = BeautifulSoup(annotation_webpage_with_attributes.content, 'lxml')
                            script = attributes_soup.find_all('script')
                            if len(attributes_soup.findAll(attrs={'id': 'aphia_attributes_group'})):
                                if 'No attributes found on lower taxonomic level' in attributes_soup.findAll(attrs={'id': 'aphia_attributes_group'})[0].getText() or len([item.getText() for item in script if 'Functional group' in item.getText()])==0:
                                    attributes_soup = ''
                                    script = ''
                    else:
                        attributes_soup=''
                        script=''
                        time.sleep(5)
                        annotation_webpage_with_attributes = session.get(urljoin(annotation_webpage.url, '#attributes'),timeout=5)  # get_attributes_output(url='https://www.marinespecies.org/aphia.php?p=search&adv=1',id=annotation_webpage.url.split('id=')[-1])
                        if annotation_webpage_with_attributes.ok:#len(list(filter(None,[id.get('id') if id.get('id') == "aphia_attributes_group_show" else id.get('href') if id.get('href') == '#attributes' else None for id in taxo_soup.findAll('a')]))) > 0:
                            attributes_soup=BeautifulSoup(annotation_webpage_with_attributes.content, 'lxml')
                            script = attributes_soup.find_all('script')
                            if len(attributes_soup.findAll(attrs={'id': 'aphia_attributes_group'})):
                                if 'No attributes found on lower taxonomic level' in attributes_soup.findAll(attrs={'id': 'aphia_attributes_group'})[0].getText() or len([item.getText() for item in script if 'Functional group' in item.getText()])==0:
                                    attributes_soup = ''
            fields = [item.getText() if len(taxo_soup.find_all(class_="alert alert-info")) == 0 else '' for item in taxo_soup.find_all(class_="col-xs-12 col-sm-4 col-lg-2 control-label")]
            Status = re.sub(r'[' + '\n' + '\xa0' + ']', '',taxo_soup.find_all(class_="leave_image_space")[fields.index('Status')].getText()) if len(taxo_soup.find_all(class_="alert alert-info")) == 0 else ''
            if Status=='' or len(re.findall(r'unaccepted', Status)) > 0: #|uncertain|unassessed
                if 'Accepted Name' not in fields:
                    continue
                else:
                    #annotation = re.sub(r'[' + '\n' + '\xa0' + ']', '', taxo_soup.find_all(class_="leave_image_space")[ fields.index('Accepted Name')].getText()) if len( taxo_soup.find_all(class_="alert alert-info")) == 0 and 'Accepted Name' in fields else annotation
                    #data['tName'] = annotation.split(' (')[0]
                    status = 'wait'
                    time.sleep(5)
                    while status!='ok':
                        taxon_list=session.get(urljoin(url, taxo_soup.find_all(class_="leave_image_space")[ fields.index('Accepted Name')].find_all('a')[0]['href']),timeout=5,stream=True)#session.post(url=url,data=data)
                        status = 'ok' if taxon_list.status_code == 200 else 'wait'
                    # Transform form query into xml table and save results in dataframe
                    time.sleep(5)
                    soup = BeautifulSoup(taxon_list.content, "lxml")
                    search_response = pd.DataFrame({'Taxon': [item.getText() for item in soup.find_all(class_="list-group-item") if any(re.findall(r'unaccepted|uncertain|unassessed',item.getText())) == False],
                                                    'Link': [item.find('a').get('href') for item in soup.find_all(class_="list-group-item") if any(re.findall(r'unaccepted|uncertain|unassessed',item.getText())) == False]})
                    if len(search_response) == 0:
                        if len(soup.find_all(class_="alert alert-info")) > 0:
                            continue
                        else:
                            taxo_soup = soup
                            attributes_soup = ''
                            script = ''
                            annotation_webpage_with_attributes = session.get(urljoin(taxon_list.url, '#attributes'),stream=True)  # get_attributes_output(url='https://www.marinespecies.org/aphia.php?p=search&adv=1',id=annotation_webpage.url.split('id=')[-1])
                            if annotation_webpage_with_attributes.ok:#len(list(filter(None,[id.get('id') if id.get('id') == "aphia_attributes_group_show" else id.get('href') if id.get('href') == '#attributes' else None for id in soup.findAll('a')]))) > 0:
                                attributes_soup = BeautifulSoup(annotation_webpage_with_attributes.content, 'lxml')
                                script = attributes_soup.find_all('script')
                                if len(attributes_soup.findAll(attrs={'id': 'aphia_attributes_group'})):
                                    if 'No attributes found on lower taxonomic level' in attributes_soup.findAll(attrs={'id': 'aphia_attributes_group'})[ 0].getText() or len([item.getText() for item in script if 'Functional group' in item.getText()]) == 0:
                                        attributes_soup = ''
                                        script = ''
                    else:
                        annotation_webpage = session.get(urljoin(url, search_response.Link[0]),timeout=5,stream=True)
                        time.sleep(5)
                        taxo_soup = BeautifulSoup(annotation_webpage.content, 'lxml')
                        attributes_soup = ''
                        annotation_webpage_with_attributes = session.get(urljoin(annotation_webpage.url, '#attributes'))  # get_attributes_output(url='https://www.marinespecies.org/aphia.php?p=search&adv=1',id=annotation_webpage.url.split('id=')[-1])
                        if annotation_webpage_with_attributes.ok: #taxo_soup.findAll('a', onclick=True)[0].get('id') == "aphia_attributes_group_show":
                            attributes_soup = BeautifulSoup(annotation_webpage_with_attributes.content, 'lxml')
                            script = attributes_soup.find_all('script')
                            if len(attributes_soup.findAll(attrs={'id': 'aphia_attributes_group'})):
                                if 'No attributes found on lower taxonomic level' in attributes_soup.findAll(attrs={'id': 'aphia_attributes_group'})[0].getText() or len([item.getText() for item in script if 'Functional group' in item.getText()])==0:
                                    attributes_soup = ''
                    fields = [item.getText() if len(taxo_soup.find_all(class_="alert alert-info")) == 0 else '' for item in taxo_soup.find_all(class_="col-xs-12 col-sm-4 col-lg-2 control-label")]

            Functional_group =[item.getText()[substring.start():]  for item in script for substring in re.finditer('Functional group', item.getText()) if 'Functional group' in item.getText()]
            Functional_group = ';'.join([str({'Functional group':group[group.find('Functional group') + 22:[sub.start() for sub in re.finditer('&nbsp', group) if sub.start() > group.find('Functional group') + 22 and sub.start()>group.find('nodes')][0]].replace(group[[sub.start() for sub in re.finditer('&nbsp', group) if sub.start() > group.find('Functional group') + 22 ][0]:[sub.start() for sub in re.finditer('&nbsp', group) if sub.start() > group.find('nodes') + 22 ][0]],''),group.split('&nbsp;" ,state: "" ,nodes: [{ text: "<b>')[1].split('<\\/b> ')[0] :group.split('&nbsp;" ,state: "" ,nodes: [{ text: "<b>')[1].split('<\\/b> ')[1].split('&nbsp')[0]}) if group.find('&nbsp;" ,state: "" ,nodes: [{ text: "<b>')!=-1 else str({'Functional group':group[group.find('Functional group') + 22:[sub.start() for sub in re.finditer('&nbsp', group) if sub.start() > group.find('Functional group') + 22][0]]}) for group in Functional_group])
            Type = np.where(taxo_soup.find_all(class_="leave_image_space")[1].getText().split('\n')[1] == 'Biota','Living', 'NA').tolist() if len( taxo_soup.find_all(class_="alert alert-info")) == 0 else ''
            dict_hierarchy = {re.sub(r'[' + string.punctuation + ']', '', level.split('\xa0')[1]): level.split('\xa0')[0] for level in taxo_soup.find_all(class_="leave_image_space")[1].getText().split('\n') if '\xa0' in level} if len( taxo_soup.find_all(class_="alert alert-info")) == 0 else dict({'': ''})
            full_hierarchy = '>'.join([level.split('\xa0')[0] + level.split('\xa0')[1] for level in taxo_soup.find_all(class_="leave_image_space")[1].getText().split('\n') if '\xa0' in level]) if len(taxo_soup.find_all(class_="alert alert-info")) == 0 else ''
            Domain = dict_hierarchy['Kingdom'] if 'Kingdom' in dict_hierarchy.keys() else ''
            Genus = dict_hierarchy['Genus'] if 'Genus' in dict_hierarchy.keys() else ''
            Family = dict_hierarchy['Family'] if 'Family' in dict_hierarchy.keys() else ''
            Order = dict_hierarchy['Order'] if 'Order' in dict_hierarchy.keys() else ''
            Class = dict_hierarchy['Class'] if 'Class' in dict_hierarchy.keys() else ''
            Phylum = dict_hierarchy['Phylum'] if 'Phylum' in dict_hierarchy.keys() else ''
            Rank = re.sub(r'[' + '\n' + ' ' + ']', '',taxo_soup.find_all(class_="leave_image_space")[fields.index('Rank')].getText()) if len(taxo_soup.find_all(class_="alert alert-info")) == 0 else ''
            URL = re.sub(r'[' + '(' + ')' + ']', '',taxo_soup.find_all(class_="aphia_core_cursor-help")[fields.index('AphiaID')].getText()) if len(taxo_soup.find_all(class_="alert alert-info")) == 0 else ''
            Original_reference = taxo_soup.find_all(class_="correctHTML")[0].getText() if len(taxo_soup.find_all(class_="correctHTML")) > 0 else ''
            Taxonomic_citation = [re.sub(r'[' + '\n' + ' ' + ']', '', item.getText()) for item in taxo_soup.find_all(class_="col-xs-12 col-sm-8 col-lg-10 pull-left") if item.attrs['id'] == 'Citation'][0] if len(taxo_soup.find_all(class_="alert alert-info")) == 0 else ''
            df_hierarchy = pd.DataFrame({'EcoTaxa_hierarchy': hierarchy,'Full_hierarchy': full_hierarchy, 'Rank': Rank,'Type': Type,'Domain': Domain,'Phylum': Phylum, 'Class': Class,'Order': Order, 'Family': Family, 'Genus': Genus,'Functional_group': Functional_group if len(Functional_group) > 0 else '','WORMS_ID': URL,'Reference': Original_reference,'Citation': Taxonomic_citation}, index=[0])
            break
    return df_hierarchy
