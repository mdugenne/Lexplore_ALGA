import cv2
import torch #pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
import imagesize
import numpy  as np
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

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