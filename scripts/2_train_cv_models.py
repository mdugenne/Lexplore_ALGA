#https://docs.ultralytics.com/datasets/segment/#dataset-yaml-format
import warnings
warnings.filterwarnings('ignore')
# Utility modules
import yaml #conda install PyYAML
import json
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path
import os

from natsort import natsorted

path_to_dir=Path('~/GIT/Lexplore_ALGA/data/datafiles/flowcam').expanduser()
os.chdir(path_to_dir)
sys.path.append(str(path_to_dir))
from sinking.funcs_image_utils import *


# Image processing modules
import numpy as np
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation
from skan import Skeleton
from scripts.funcs_image_processing import *
from sklearn.model_selection import train_test_split

# Segmentation module
from ultralytics import YOLO

# Workflow starts here

path_to_cnn=Path(str(path_to_ecotaxa_flowcam_files).replace(os.sep+'ecotaxa',os.sep+'cnn')).expanduser()
IMAGE_SIZE=640
pixel_size=0.7339# in pixels per micrometer # cfg_metadata['pixel_size_flowcam_10x']

############################# Instance segmentation ############################
# Prepare dataset directories and files for instance segmentation
path_to_train=path_to_cnn / "instance_segmentation" /'train'
path_to_validation=path_to_cnn / "instance_segmentation" /'validation'

path_to_train.mkdir(parents=True,exist_ok=True)
path_to_validation.mkdir(parents=True,exist_ok=True)

path_to_ecotaxa_export=path_to_git / 'data' / 'datafiles' / 'ecotaxa'

file_ecotaxa_export=natsorted(list(path_to_ecotaxa_export.glob('ecotaxa_14791_export_annotations_*.csv')))[0]
df_annotations=pd.read_csv(file_ecotaxa_export).dropna(subset=['object_annotation_name']).query('object_annotation_lineage.str.contains("temporary", na=False,case=False)==False').reset_index(drop=True)
df_annotations['class_name']=np.where(df_annotations.object_annotation_name.str.lower().str.split('<').str[0].str.split(' ').str[0].str.split('-').str.len()>1,df_annotations.object_annotation_name.str.lower().str.split('<').str[0].str.split(' ').str[0].str.split('-').str[1],df_annotations.object_annotation_name.str.lower().str.split('<').str[0].str.split(' ').str[0].str.split('-').str[0])
df_annotations.query('class_name=="chytrid"').drop_duplicates(subset=['object_annotation_name'], keep='first').object_annotation_name.unique()

#Ensure image have been saved during step 0
df_thumbnails=pd.DataFrame({'image_path':natsorted(list(Path(path_to_cnn /'input' ).expanduser().glob('Flowcam_10x_*/*.jpg')))})
df_thumbnails['image_id']=df_thumbnails.image_path.astype(str).str.split(os.sep).str[-1]
df_annotations=df_annotations[(df_annotations.object_filename.isin(df_thumbnails.image_id.unique())==True)].reset_index(drop=True)
df = ((df_annotations.query('class_name=="chytrid"').assign(class_id=0,image_id=lambda x: x.object_filename))[['image_id','class_name','class_id','object_annotation_name']]).reset_index(drop=True)
df=pd.merge(df,df_thumbnails,how='left',on='image_id')

SEED=5
df_train, df_valid = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
path_to_train=path_to_git / 'datasets' / 'instance_segmentation' /'images' / 'train'
path_to_validation=path_to_git / 'datasets' / 'instance_segmentation'/'images' / 'val'
df=pd.concat([df_train.assign(path_to_storage=lambda x: path_to_train / x.image_id),df_valid.assign(path_to_storage=lambda x: path_to_validation / x.image_id)],axis=0).reset_index(drop=True)

#Generate masks for segmentation and extract their coordinates for computer vision model
Min_ESD,Max_ESD=15,95
idx=np.random.randint(len(df))
path=df.loc[idx,'image_path']
image=df.loc[idx,'image_id']

df_checked=pd.DataFrame({'image_path':natsorted(list(Path(path_to_cnn /'instance_segmentation'  ).expanduser().glob('validation/images/*.jpg')))})
df_checked=pd.concat([df_checked,pd.DataFrame({'image_path':natsorted(list(Path(path_to_cnn /'instance_segmentation'  ).expanduser().glob('train/images/*.jpg')))})],axis=0).reset_index(drop=True)
df_checked['image_id']=df_checked.image_path.astype(str).str.split(os.sep).str[-1]

df=df[df.image_id.isin(df_checked.image_id)==False]

with tqdm(desc='Generating formatted vignettes for {}'.format(image), total=len(df), bar_format='{desc}{bar}', position=0, leave=True) as bar:
    #idx,path=1,df.loc[1,'image_path']
    for idx,path in df['image_path'].items():
        bar.set_description('Generating formatted vignettes for {}'.format(path.stem), refresh=True)
        colored_image=image = cv2.imread(str(path))
        #plt.figure(), plt.imshow(colored_image),plt.show()

        gray_image=image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #plt.figure(),plt.imshow(gray_image)


        # Segmentation
        # Watershed segmentation
        edges = ski.filters.sobel((colored_image[:,:,1]<125) )#ski.filters.sobel(image) #& (colored_image[:,:,0]<125) & (colored_image[:,:,2]<125)
        #mask =ski.morphology.dilation(ski.morphology.erosion(ski.morphology.closing(edges),skimage.morphology.disk(2)),skimage.morphology.disk(1))# ski.morphology.closing(ski.morphology.dilation(cv2.cvtColor(markers, cv2.COLOR_BGR2GRAY),skimage.morphology.disk(3))) #cv2.cvtColor(markers, cv2.COLOR_BGR2GRAY)
        markers = np.zeros_like(image)
        markers[edges >= 0.9 * edges.max()] = 1
        markers[(sp.ndimage.binary_closing(edges) == False)] = 0
        mask = ski.morphology.closing(ski.morphology.dilation(markers, skimage.morphology.disk( 3)))  # ski.morphology.closing(ski.morphology.dilation(cv2.cvtColor(markers, cv2.COLOR_BGR2GRAY),skimage.morphology.disk(3))) #cv2.cvtColor(markers, cv2.COLOR_BGR2GRAY)

        label_objects, nb_labels = sp.ndimage.label(mask)
        # plt.figure(), plt.imshow(label_objects),plt.show()
        label_objects = ski.morphology.remove_small_objects(label_objects, min_size=int(5 / pixel_size))
        #label_objects = label_objects ^ ski.morphology.remove_small_objects(label_objects,min_size=int(Max_ESD / pixel_size))
        label_chl_objects, nb_chl_labels= sp.ndimage.label(label_objects)
        #label_chl_objects, nb_chl_labels = sp.ndimage.label(mask)
        # plt.figure(), plt.imshow(label_chl_objects),plt.show()
        df_chl_coords = pd.DataFrame(ski.measure.regionprops_table(label_image=label_chl_objects.astype('int32'), properties=['label', 'area', 'area_bbox','bbox', 'centroid', 'slice']))

        # Remove chloroplast inside host
        edges =skimage.filters.sobel(ski.filters.gaussian(image,1))
        # plt.figure(),plt.imshow(edges),plt.show()
        mask = (ski.morphology.closing(ski.morphology.erosion(ski.morphology.dilation(edges>np.quantile(edges,0.92), skimage.morphology.disk(4)),skimage.morphology.disk(4))))
        # plt.figure(),plt.imshow(mask),plt.show()
        label_objects, nb_labels = sp.ndimage.label(mask)
        label_objects=ski.morphology.remove_small_objects(label_objects, min_size=int(Min_ESD/pixel_size))
        label_objects, nb_labels = sp.ndimage.label(label_objects)
        # plt.figure(),plt.imshow(label_objects),plt.show()
        labelled_particle=measure.label(label_objects.astype(bool).astype(int))
        label_objects_of_interest=np.isin(labelled_particle,[label for label in np.unique(labelled_particle) if label not in [0,np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1]])
        # plt.figure(),plt.imshow(label_objects_of_interest),plt.show()

        label_objects = ski.morphology.erosion(label_objects, skimage.morphology.disk(2))
        labelled_particle = measure.label(label_objects.astype(bool).astype(int))
        label_objects=labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1
        label_objects, nb_labels = sp.ndimage.label(label_objects)
        df_roi_coords = pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects.astype('int32'), properties=['label', 'area', 'area_bbox','bbox', 'centroid', 'slice','axis_minor_length','feret_diameter_max','equivalent_diameter_area']))
        df_roi_polygon=pd.DataFrame(skimage.measure.find_contours(label_objects,level=0.0)[0],columns=['x','y'])

        skeleton_mask = skimage.morphology.skeletonize(ski.morphology.closing(ski.morphology.erosion(ski.morphology.dilation(edges>0.05, skimage.morphology.disk(4)),skimage.morphology.disk(5))),method='lee')
        #skeleton_mask = skimage.morphology.thin(ski.morphology.closing( ski.morphology.erosion(ski.morphology.dilation(edges > 0.05, skimage.morphology.disk(4)), skimage.morphology.disk(5))))

        # plt.figure(),plt.imshow(skeleton_mask),plt.show()
        if len(skeleton_mask[skeleton_mask == True]) > 2:
            skeleton = Skeleton(skeleton_mask)
            end_points = skeleton.coordinates[skeleton.degrees == 1]
            #  plt.figure(), plt.imshow(image, cmap='gray'),plt.scatter(np.moveaxis(end_points,1,0)[1],np.moveaxis(end_points,1,0)[0],c='red',marker='o'), plt.show()
            #  plt.scatter(df_roi_polygon.y, df_roi_polygon.x,s=0.9, color='black',linewidths=0.1)
            end_points_check=ski.measure.points_in_poly(end_points,df_roi_polygon[['x','y']])
            if any(end_points_check==False):

                marker=np.zeros_like(mask)
                for point in range(len(end_points[end_points_check == False])):
                    # Get the coordinates of the disk
                    rr, cc = ski.draw.disk(tuple(end_points[end_points_check == False][point]), radius=df_roi_coords.at[0,'equivalent_diameter_area']/10, shape=image.shape)
                    # Set the pixels within the disk to True in the mask
                    marker[rr, cc] = 1

                #plt.figure(), plt.imshow(marker), plt.show()

                label_objects_of_interest_chl,nb_of_interest_labels= sp.ndimage.label(marker)
                # plt.figure(), plt.imshow(label_objects_of_interest_chl), plt.show()
            else:
                label_objects_of_interest_chl, nb_of_interest_labels = sp.ndimage.label(np.zeros_like(mask))
        else:
            label_objects_of_interest_chl, nb_of_interest_labels = sp.ndimage.label(label_chl_objects)
            label_objects_of_interest, nb_of_interest_labels = sp.ndimage.label(np.zeros_like(mask))

        #label_objects_of_interest_chl=np.isin(measure.label(label_chl_objects.astype(bool).astype(int)),[roi for roi in np.unique(skimage.segmentation.join_segmentations(label_objects,label_chl_objects)) if (roi in range(nb_chl_labels)[1:])])
        # plt.figure(), plt.imshow(label_objects_of_interest_chl),plt.show()
        label_objects, nb_labels= sp.ndimage.label(skimage.segmentation.join_segmentations(label_objects_of_interest.astype(int),label_objects_of_interest_chl.astype(int)))
        # plt.figure(), plt.imshow(label_objects),plt.show()
        #skimage.segmentation.join_segmentations(label_objects, label_chl_objects,return_mapping=True)
        #plt.figure(),plt.imshow(skimage.segmentation.join_segmentations(label_objects,label_chl_objects),cmap='twilight'),plt.show()
        #plt.figure(), plt.imshow(skimage.segmentation.join_segmentations(label_objects, label_chl_objects) > nb_chl_labels), plt.show()


        '''
        # Segmentation
        # Watershed segmentation
        edges = ski.filters.sobel((colored_image[:,:,1]<125) )#ski.filters.sobel(image)
        #plt.figure(),plt.imshow(edges , cmap='gray'),plt.show()
        markers = np.zeros_like(image)
        markers[edges >= 0.9*edges.max()] = 1
        markers[(sp.ndimage.binary_closing(edges) == False)] = 0
        mask = ski.morphology.closing(ski.morphology.dilation(markers, skimage.morphology.disk( 3)))  # ski.morphology.closing(ski.morphology.dilation(cv2.cvtColor(markers, cv2.COLOR_BGR2GRAY),skimage.morphology.disk(3))) #cv2.cvtColor(markers, cv2.COLOR_BGR2GRAY)

        #plt.figure(),plt.imshow(mask),plt.show()
        label_objects, nb_labels = sp.ndimage.label(mask)
        label_objects=ski.morphology.remove_small_objects(label_objects, min_size=int(Min_ESD/pixel_size))
        label_objects = label_objects ^ ski.morphology.remove_small_objects(label_objects, min_size=int(Max_ESD/pixel_size))
        label_objects, nb_labels = sp.ndimage.label(label_objects)
        #plt.figure(), plt.imshow(label_objects),plt.show()
        '''

        unscaled_objects=label_objects


        if nb_labels>0:
            # Resize image to fit model input image shape while keeping aspect ratio
            label_objects=pad(resize_to_square(label_objects.astype('float32'), size=np.max(gray_image.shape)), IMAGE_SIZE,IMAGE_SIZE )
            label_objects[np.isin(label_objects,np.unique(unscaled_objects))==False]=0
            if len(skimage.measure.find_contours(label_objects,level=0.0)):
                df_coordinates=pd.concat(list(map(lambda array:pd.DataFrame(np.ceil(array[1]),columns=['x','y']).assign(label=array[0]+1),enumerate(skimage.measure.find_contours(label_objects,level=0.0)))))
                df_coordinates=pd.merge(df_coordinates,pd.DataFrame(ski.measure.regionprops_table(label_image=label_objects.astype('int32'),properties=['label','area', 'area_bbox','bbox', 'centroid', 'slice'])),how='left',on='label')
                padded_image = pad(resize_to_square(colored_image, size=np.max(gray_image.shape)), IMAGE_SIZE, IMAGE_SIZE)

                # Save segmented thumbnail for CV model
                save_file = df.loc[idx, 'path_to_storage']
                save_file.parent.mkdir(parents=True, exist_ok=True)
                save_mask=str(save_file).replace(os.sep + 'images' + os.sep, os.sep + 'labelled' + os.sep)
                Path(save_mask).parent.mkdir(parents=True, exist_ok=True)

                plt.ioff()  # Turn off interactive mode to hide the plot
                fig, axes = plt.subplots(1, 1,frameon=False)
                plt.imshow(padded_image, cmap='gray')
                axes.set_axis_off()
                df_coords=pd.DataFrame()
                for label in df_coordinates.label.unique():
                    df_label=df_coordinates.query('label=={}'.format(label)).drop(columns=['x','y']).drop_duplicates()
                    #axes.add_patch(Rectangle(fill=False,xy=(df_label['bbox-1'].values[0],df_label['bbox-0'].values[0]), width=(df_label['bbox-3'].values[0]-df_label['bbox-1'].values[0]), height=(df_label['bbox-2'].values[0]-df_label['bbox-0'].values[0])))
                    axes.add_line(Line2D(df_coordinates.query('label=={}'.format(label)).y,df_coordinates.query('label=={}'.format(label)).x))
                    #fig.show()
                    # Normalize coordinates
                    x_coords, y_coords = (df_coordinates.query('label=={}'.format(label)).y ) / IMAGE_SIZE, (df_coordinates.query('label=={}'.format(label)).x) / IMAGE_SIZE
                    list_coords = [int(df.loc[idx, 'class_id'])] + list(sum(zip(x_coords.tolist(), y_coords.tolist() + [0]), ())[:])
                    df_coords =pd.concat([df_coords,pd.DataFrame(list_coords).T.astype({0:int})],axis=0,ignore_index=True).astype({0:int})
                fig.savefig(fname=str(save_file).replace(os.sep+'images'+os.sep,os.sep+'labelled'+os.sep), transparent=False, bbox_inches="tight", pad_inches=0, dpi=300)
                plt.close('all')

                # Save xy coordinates
                save_label = str(save_file).replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).replace('.jpg','.txt')
                Path(save_label).parent.mkdir(parents=True, exist_ok=True)
                df_coords.to_csv(save_label,sep=' ',index=False,header=False,lineterminator=os.linesep)




                # Save raw thumbnail for CV model
                colored_image=pad(resize_to_square(colored_image, size=np.max(gray_image.shape)), IMAGE_SIZE,IMAGE_SIZE )
                resized_thumbnail = Image.fromarray(colored_image)
                resized_thumbnail.save(save_file)



        bar.update(1)

# Train pretrained (COCO) model on custom datasets generated before

# open json configuration file to change the saving directory

path_to_config = Path.home()/'AppData'/'Roaming'/'Ultralytics'/'settings.json'
with open(path_to_config, 'r') as config_file:
    cfg_data = json.load(config_file)

cfg_data['datasets_dir']=str(path_to_git / 'models')
cfg_data['runs_dir']='models'
json_str = json.dumps(cfg_data, indent=2)
with open(path_to_config, "w") as f:
    f.write(json_str)

from ultralytics.data.converter import convert_coco

convert_coco(labels_dir=path_to_cnn / "instance_segmentation" /'labels', use_segments=True)
# Load the model.
model = YOLO("yolo11l-seg.pt")  # load a pretrained model (recommended for training)
yml_file=path_to_cnn / "instance_segmentation" / "yolo_instance_segmentation.yaml"
# Train the model
EPOCHS=100
path_to_storage=Path('models' / "instance_segmentation" / 'models'/ 'yolo_instance_segmentation_custom_')
results = model.train(
    save_dir=path_to_storage,
    save_txt=,
    save_json=,
    data=yml_file,
    imgsz=640,
    epochs=EPOCHS,
    batch=8,
    lr0=LR,
    name='yolo_instance_segmentation_custom')

# Validate the model

model = YOLO(str(path_to_cnn / "instance_segmentation" / 'yolo_instance_segmentation_custom_{}'.format(str(int(EPOCHS))) / 'weight'/ "best.pt"))  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list containing mAP50-95(B) for each category
metrics.seg.map  # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps  # a list containing mAP50-95(M) for each category