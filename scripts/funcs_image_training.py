#https://towardsdatascience.com/an-in-depth-efficientnet-tutorial-using-tensorflow-how-to-use-efficientnet-on-a-custom-dataset-1cab0997f65c/
import warnings
warnings.filterwarnings('ignore')

import keras
from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from tensorflow.keras import optimizers
import tensorflow as tf
#Use this to check if the GPU is configured correctly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

try:
    from scripts.funcs_image_processing import *
except:
    from funcs_image_processing import *

from natsort import natsorted

from scripts.funcs_image_utils import CNNDataset, create_dataloader


#input_shape is (height, width, number of channels) for images
input_shape=(224, 224, 3)

path_to_ecotaxa_export=path_to_git / 'data' / 'datafiles' / 'ecotaxa'
path_to_cnn=Path(path_to_network /'Imaging_Flowcam' / 'Flowcam data' / 'Lexplore' / 'cnn').expanduser()

file_ecotaxa_export=natsorted(list(path_to_ecotaxa_export.glob('ecotaxa_14791_export_annotations_*.csv')))
if len(file_ecotaxa_export)==0:
    export_ecotaxa_annotations(configuration=configuration,project_id=str(cfg_metadata['ecotaxa_lexplore_alga_flowcam_micro_projectid']),export_path=path_to_ecotaxa_export)

file_ecotaxa_export=natsorted(list(path_to_ecotaxa_export.glob('ecotaxa_14791_export_annotations_*.csv')))[0]
df_annotations=pd.read_csv(file_ecotaxa_export).dropna(subset=['object_annotation_name']).query('object_annotation_lineage.str.contains("temporary", na=False,case=False)==False').reset_index(drop=True)
df_annotations['object_annotation_name']=df_annotations.object_annotation_name.str.lower().str.split('<').str[0].str.split(' ').str[0].str.split('-').str[0]
natsorted(df_annotations.object_annotation_name.unique())
df_annotations.object_annotation_name.nunique()
# Ensure thumbnail exists
MIN_CLASS=5
df_thumbnails=pd.DataFrame({'image_path':natsorted(list(Path(path_to_cnn / 'input').glob('Flowcam*/*.jpg')))})
df_thumbnails['image_id']=df_thumbnails.image_path.astype(str).str.split(os.sep).str[-1]
df_annotations=df_annotations[(df_annotations.object_filename.isin(df_thumbnails.image_id.unique())==True)].reset_index(drop=True)
df_annotations=df_annotations[(df_annotations.object_annotation_name.isin(df_annotations.object_annotation_name.value_counts().to_frame().query('count<={}'.format(MIN_CLASS)).index)==False)].reset_index(drop=True)

classes = natsorted(df_annotations.object_annotation_name.unique())
dict_classes=dict(zip(classes,range(len(classes))))
df_classes=pd.DataFrame(dict(zip(range(len(classes)),classes)),index=[0]).T
df = ((df_annotations.assign(image_id=lambda x: x.object_filename,class_name=lambda x: x.object_annotation_name)).assign(class_id=lambda x: x.class_name.map(dict_classes)).assign(label=lambda x: x.class_id))[['image_id','class_name','class_id','label']]
df=pd.merge(df,df_thumbnails,how='left',on='image_id')
#df.to_csv(str(path_to_cnn /'dataset_train_test.csv'),index=False)
#df=pd.read_csv(str(path_to_cnn /'dataset_train_test.csv'))

NUMBER_OF_CLASSES=df.class_id.nunique() # 78
SEED=5
classes = natsorted(df.class_name.unique())
dict_classes=dict(zip(classes,range(len(classes))))
df_classes=pd.DataFrame(dict(zip(range(len(classes)),classes)),index=[0]).T

TRAIN_IMAGES_PATH = path_to_cnn /'input' / 'train'
VAL_IMAGES_PATH = path_to_cnn/'input' / 'validation'
TRAIN_IMAGES_PATH.mkdir(parents=True,exist_ok = True)
VAL_IMAGES_PATH.mkdir(parents=True,exist_ok = True)

conv_base = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=input_shape)
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
#avoid overfitting
model.add(layers.Dropout(rate=0.2, name="dropout_out"))
# Set NUMBER_OF_CLASSES to the number of your final predictions.
model.add(layers.Dense(NUMBER_OF_CLASSES, activation="softmax", name="fc_out"))
conv_base.trainable = False


# Create directories for each class.
for class_id in [x for x in range(len(classes))]:
    os.makedirs(os.path.join(TRAIN_IMAGES_PATH, str(class_id)), exist_ok = True)
    os.makedirs(os.path.join(VAL_IMAGES_PATH, str(class_id)), exist_ok = True)


Input_dir = path_to_cnn
def organize_class_images(df,images_path):
    for column, row in tqdm(df.iterrows(), total=len(df)):
        class_id = row['class_id']
        shutil.copy(src=row['image_path'], dst=os.path.join(images_path, str(class_id)))

def organize_class_images_pytorch(df,images_path):
    for column, row in tqdm(df.iterrows(), total=len(df)):
        class_id = row['class_id']
        shutil.copy(src=row['image_path'], dst=os.path.join(images_path))

#Split the dataset into 80% training and 20% validation
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
df_train=df[df.class_id.astype(str).isin(df.class_id.astype(str).value_counts().to_frame().query('count==1').index)].reset_index(drop=True)
df_valid =pd.DataFrame()
df_subset=df.copy()
df_subset=df_subset[df_subset.class_id.astype(str).isin(df_subset.class_id.astype(str).value_counts().to_frame().query('count==1').index)==False].reset_index(drop=True)
natsorted(df_subset.class_id.unique())
for train_idx, test_idx in splitter.split(df_subset.image_id,df_subset.class_id):
    try:
        df_train=pd.concat([df_train,df_subset.loc[train_idx]],axis=0,ignore_index=True).reset_index(drop=True)
        df_valid=pd.concat([df_valid,df_subset.loc[test_idx]],axis=0,ignore_index=True).reset_index(drop=True)
    except:
        df_train=df_train
        df_valid=df_valid

#df_train, df_valid = model_selection.train_test_split(df, test_size=0.2, random_state=5, shuffle=True)
df_train.class_id.nunique(),df_valid.class_id.nunique()
df_train.class_name.value_counts()

#run the function on each of them
organize_class_images(df_train, TRAIN_IMAGES_PATH)
organize_class_images(df_valid, VAL_IMAGES_PATH)

organize_class_images_pytorch(df_train, TRAIN_IMAGES_PATH)
organize_class_images_pytorch(df_valid, VAL_IMAGES_PATH)

# Use ImageDataGenerator to perform data augmentation
from keras.layers import Resizing
def preprocess(image):
    image = np.array(image)
    transformed_img=np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]*3,axis=2)
    transformed_img=pad(resize_to_square(transformed_img, size=np.max(image.shape)), input_shape[0], input_shape[1])
    return transformed_img

train_datagenerator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    preprocessing_function=preprocess,#keras.layers.Resizing(height=input_shape[0],width=input_shape[1],crop_to_aspect_ratio=True,pad_to_aspect_ratio=True),
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagenerator = ImageDataGenerator(rescale=1.0 / 255,
                                        preprocessing_function=preprocess#keras.layers.Resizing(height=input_shape[0],width=input_shape[1],crop_to_aspect_ratio=True,pad_to_aspect_ratio=True)
                                        )
BATCH_SIZE=8
train_generator = train_datagenerator.flow_from_directory(
    # This is the target directory
    directory=TRAIN_IMAGES_PATH,
    # All images will be resized to target height and width.
    target_size=(input_shape[0], input_shape[1]),
    batch_size=BATCH_SIZE,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode="categorical",keep_aspect_ratio=True,
)

validation_generator = test_datagenerator.flow_from_directory(
    directory=VAL_IMAGES_PATH,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=BATCH_SIZE,
    class_mode="categorical",keep_aspect_ratio=True,
)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=LR),
    metrics=["accuracy"],
)

# Train the model
EPOCHS = 15
model_cnn =model.fit(train_generator,
    steps_per_epoch=train_generator.n // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // BATCH_SIZE,
    verbose=1)


# Predictions on validation
x,y=next(iter(validation_generator))
image = x[0, :, :, :]
true_index = np.argmax(y[0])
plt.imshow(image)
plt.axis('off')
plt.show()

prediction_scores = model.predict(np.expand_dims(image, axis=0))
predicted_index = np.argmax(prediction_scores)
print("True label: " + classes[true_index])
print("Predicted label: " + classes[predicted_index])

y_pred = model.predict(validation_generator)
df_pred=pd.DataFrame(y_pred,columns=classes)
np.argmax(df_pred,axis=1)
score = model.evaluate(validation_generator,verbose=1)


## Pytorch
#https://www.kaggle.com/code/shnakazawa/image-classification-with-pytorch-and-efficientnet/notebook

import os
import numpy as np
import pandas as pd
import math
import time
import random
import gc
import cv2
from pathlib import Path
from tqdm import tqdm


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

RUN_TRAINING = True
TRAIN_ALL = False # If true, train with all data and output a single model. If False, run cross-validation and output multiple models.
FOLD_NUM = 3 # For cross-validation
EPOCHS = 5 # Training cycle
RUN_INFERENCE = False
NUM_CLASSES=NUMBER_OF_CLASSES

# Directory setting
DATA_DIR = path_to_cnn
MODEL_DIR = path_to_cnn
CSV_SAVE_DIR = path_to_cnn
IMG_SAVE_DIR = path_to_cnn
#for file in df.image_path.unique():
#    if Path(path_to_cnn /'input' /'train_all'/file.split(os.sep)[-1]).exists()==False:
#        shutil.copy(src=file, dst=path_to_cnn /'input' /'train_all')

# Ensure all images are present in training directory
path_to_train_directory=path_to_cnn /'input' / 'train_all'
for i,file in df.image_path.items():
    if Path(path_to_cnn /'input' /'train_all'/file.split(os.sep)[-1]).exists()==False:
        df=df.drop(index=i).reset_index(drop=True)


# Cross-validation for model training
if RUN_TRAINING and (TRAIN_ALL == False):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Cross-validation
    folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED).split(np.arange(df.shape[0]), df['class_id'].to_numpy())

    # For Visualization of Accuracy and Loss metrics
    train_acc_list = []
    valid_acc_list = []
    train_loss_list = []
    valid_loss_list = []

    for fold, (trn_idx, val_idx) in enumerate(folds):
        print(f'==========Cross Validation Fold {fold + 1}==========')
        # Load Data
        train_loader, valid_loader = create_dataloader(df, trn_idx, val_idx,train_path=path_to_train_directory)

        # Load model, loss function, and optimizing algorithm
        model = EfficientNet_V2(NUM_CLASSES).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # For Visualization
        train_accs = []
        valid_accs = []
        train_losses = []
        valid_losses = []

        # Start training
        best_acc = 0
        #epoch=0
        for epoch in range(EPOCHS):
            time_start = time.time()
            #print(f'==========Epoch {epoch + 1} Start Training==========')
            model.train()

            epoch_loss = 0
            epoch_accuracy = 0

            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for step, (img, label) in pbar:
                img = img.to(device).float()
                label = label.to(device).long()

                output = model(img)
                loss = loss_fn(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)

            #print(f'==========Epoch {epoch + 1} Start Validation==========')
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                val_labels = []
                val_preds = []

                pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
                for step, (img, label) in pbar:
                    img = img.to(device).float()
                    label = label.to(device).long()

                    val_output = model(img)
                    val_loss = loss_fn(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(valid_loader)
                    epoch_val_loss += val_loss / len(valid_loader)

                    val_labels += [label.detach().cpu().numpy()]
                    val_preds += [torch.argmax(val_output, 1).detach().cpu().numpy()]

                val_labels = np.concatenate(val_labels)
                val_preds = np.concatenate(val_preds)

            # print result from this epoch
            exec_t = int((time.time() - time_start) / 60)
            print( f'Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} / Exec time {exec_t} min\n')

            # For visualization
            train_accs.append(epoch_accuracy.cpu().numpy())
            valid_accs.append(epoch_val_accuracy.cpu().numpy())
            train_losses.append(epoch_loss.detach().cpu().numpy())
            valid_losses.append(epoch_val_loss.detach().cpu().numpy())

        train_acc_list.append(train_accs)
        valid_acc_list.append(valid_accs)
        train_loss_list.append(train_losses)
        valid_loss_list.append(valid_losses)
        del model, optimizer, train_loader, valid_loader, train_accs, valid_accs, train_losses, valid_losses
        gc.collect()
        torch.cuda.empty_cache()

    show_validation_score(train_acc_list, train_loss_list, valid_acc_list, valid_loss_list)

else:
    print('Cross validation is not performed')

# Use cross-validation best hyperparameters to train the final model
RUN_TRAINING = True
TRAIN_ALL = True
EPOCHS=15
df_metrics=pd.DataFrame()
if RUN_TRAINING and TRAIN_ALL:
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    train_datasets = CNNDataset(df, path_to_train_directory, transforms=transform_train())
    train_loader = DataLoader(train_datasets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    # Load model, loss function, and optimizing algorithm
    model = EfficientNet_V2(NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Start training
    for epoch in range(EPOCHS):
        time_start = time.time()
        #print(f'==========Epoch {epoch + 1} Start Training==========')
        model.train()

        epoch_loss = 0
        epoch_accuracy = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (img, label) in pbar:
            img = img.to(device).float()
            label = label.to(device).long()

            output = model(img)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
        df_metrics=pd.concat([df_metrics,pd.DataFrame({'epochs':epoch,'accuracy':torch.Tensor.tolist(epoch_accuracy),'loss':torch.Tensor.tolist(epoch_loss)},index=[0])],axis=0).reset_index(drop=True)
        df_metrics.to_csv(str(MODEL_DIR / 'model_metrics.csv'),index=False)
        # print results from this epoch
        exec_t = int((time.time() - time_start) / 60)
        print(f'Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} / Exec time {exec_t} min\n')

        print(f'Save model trained with all data')
        MODEL_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), str(MODEL_DIR / 'classification_efficientnet_v2_s.pth'))

    del model, optimizer, train_loader
    gc.collect()
    torch.cuda.empty_cache()

else:
    print('Training with all data is not performed')


# Model predictions on test set
RUN_INFERENCE = True
if RUN_INFERENCE:
    df_validation = df_valid.drop(columns=['label'])
    df_validation.class_id.nunique()
    #df_validation['image_id'] =[path.name for path in  list(Path(path_to_cnn/ 'input' /'validation').glob('*.jpg') )]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    test_datasets = CNNDataset(df_validation, str(path_to_cnn/ 'input' /'validation'), transforms=transform_valid(), give_label=False)

    # Data Loader
    test_loader = DataLoader(test_datasets, batch_size=1, num_workers=0, shuffle=False)

    # Load model, loss function, and optimizing algorithm
    model = EfficientNet_V2(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(str(MODEL_DIR / 'classification_efficientnet_v2_s.pth')))

    # Start Inference
    print(f'Start Inference')
    with torch.no_grad():
        test_preds = []
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for step, img in pbar:
            img = img.to(device).float()
            test_output = model(img)
            test_preds += [torch.argmax(test_output, 1).detach().cpu().numpy()]
        test_preds = np.concatenate(test_preds)
    df_validation['label'] = test_preds
    df_validation.head()
    df_validation.to_csv('classification_efficientnet_v2_s_validation.csv', index=False)
    print(f'Inference saved')
else:
    print('RUN_INFERENCE is False')

#df_validation=pd.merge(df_validation,df.drop(columns=['label']),on='image_id',how='left')
df_validation.class_id.value_counts()
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, r2_score
accuracy_score(df_validation['class_id'], df_validation['label'])
f1_score(df_validation['class_id'], df_validation['label'],average='weighted')
r2_score(df_validation['class_id'], df_validation['label'])
precision_score(df_validation['class_id'], df_validation['label'],average='weighted')
recall_score(df_validation['class_id'], df_validation['label'],average='weighted')
df_confusion=confusion_matrix(df_validation['class_id'], df_validation['label'])
df_confusion.sum(axis=1)
df_validation.class_id.astype(float).value_counts().to_frame().reset_index().sort_values(by=['class_id'])
df_confusion=pd.DataFrame(df_confusion/np.where(np.stack([df_confusion.sum(axis=1)]*df_confusion.shape[0],axis=1)==0,1,np.stack([df_confusion.sum(axis=1)]*df_confusion.shape[0],axis=1)))
df_confusion.index=df_confusion.index.map(dict(zip(dict_classes.values(),dict_classes.keys())))
df_confusion.columns=df_confusion.columns.map(dict(zip(dict_classes.values(),dict_classes.keys())))
df_confusion.to_csv(str(MODEL_DIR / 'classification_efficientnet_v2_s_confusion_matrix.csv'),index=False)