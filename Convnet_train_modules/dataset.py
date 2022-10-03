import os
import cv2
import math
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from torchvision import transforms
try:
    from torchvision.transforms.functional import InterpolationMode
    has_interpolation_mode = True
except ImportError:
    has_interpolation_mode = False

from albumentations import (
    Compose,
    GaussianBlur,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    Resize,
    ShiftScaleRotate,
    VerticalFlip,
    RandomBrightnessContrast,
    ImageCompression,
    Cutout
)
from albumentations.pytorch.transforms import ToTensorV2

DEBUG = False
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

if hasattr(Image, "Resampling"):
    _pil_interpolation_to_str = {
        Image.Resampling.NEAREST: 'nearest',
        Image.Resampling.BILINEAR: 'bilinear',
        Image.Resampling.BICUBIC: 'bicubic',
        Image.Resampling.BOX: 'box',
        Image.Resampling.HAMMING: 'hamming',
        Image.Resampling.LANCZOS: 'lanczos',
    }
else:
    _pil_interpolation_to_str = {
        Image.NEAREST: 'nearest',
        Image.BILINEAR: 'bilinear',
        Image.BICUBIC: 'bicubic',
        Image.BOX: 'box',
        Image.HAMMING: 'hamming',
        Image.LANCZOS: 'lanczos',
    }
_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}


if has_interpolation_mode:
    _torch_interpolation_to_str = {
        InterpolationMode.NEAREST: 'nearest',
        InterpolationMode.BILINEAR: 'bilinear',
        InterpolationMode.BICUBIC: 'bicubic',
        InterpolationMode.BOX: 'box',
        InterpolationMode.HAMMING: 'hamming',
        InterpolationMode.LANCZOS: 'lanczos',
    }
    _str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}
else:
    _pil_interpolation_to_torch = {}
    _torch_interpolation_to_str = {}


def str_to_interp_mode(mode_str):
    if has_interpolation_mode:
        return _str_to_torch_interpolation[mode_str]
    else:
        return _str_to_pil_interpolation[mode_str]
    

def create_folds(df):
    skf_kwargs = {'X':df, 'y':df['label']}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    df['fold'] = -1
    for fold_id, (train_idx, valid_idx) in enumerate(skf.split(**skf_kwargs)):
        df.loc[valid_idx, "fold"] = fold_id

    classes = sorted(df['label'].unique())
    label_mapping = dict(zip(classes, list(range(len(classes)))))
    return df, label_mapping


class DatasetRetriever(torch.utils.data.Dataset):
    def __init__(self, data_path, dataframe, transform=None, label_mapping=None):
        self.data_path = data_path
        self.dataframe = dataframe.reset_index(drop=True)
        if DEBUG:
            self.dataframe = self.dataframe[:35000]
        self.transform = transform 
        self.label_mapping = label_mapping
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, item):

        # image = cv2.imread(f"raw/{record.label}/{record.image_name}", cv2.IMREAD_COLOR).astype(np.float32)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image / 255.0 
        # if self.transform:
        #     sample = {
        #         'image': image
        #     }
        #     image = self.transform(**sample)['image']
        
        row = self.dataframe.iloc[item]
        image_path = os.path.join(self.data_path, row.label, row.image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.label_mapping[row.label]
        label = torch.tensor(label)
        
        return {
            'image':image, 
            'labels':label
        }


def generate_transforms(image_size):
    train_transform = Compose(
        [
            Resize(image_size[0], image_size[1], p=1.0),
            HorizontalFlip(p=0.5),
            ImageCompression(quality_lower=99, quality_upper=100),
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
            Cutout(max_h_size=int(image_size[0] * 0.4), max_w_size=int(image_size[1] * 0.4), num_holes=1, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ToTensorV2()
        ]
    )

    val_transform = Compose(
        [
            Resize(height=image_size[0], width=image_size[1], p=1.0),
            ToTensorV2()
        ]
    )

    return {"train_transforms": train_transform, "validation_transforms": val_transform}
    

def transforms_train(image_size, interpolation='bilinear', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    if interpolation == 'random':
        interpolation = 'bilinear'
    tfl = [
        transforms.Resize(image_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(image_size)
    ]
    tfl += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ]
    return transforms.Compose(tfl)


def transforms_validation(
    image_size=224, crop_pct=None, interpolation='bilinear', 
    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    scale_size = int(math.floor(image_size / crop_pct))
    tfl = [
        transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(image_size),
    ]
    tfl += [
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
    ]
    return transforms.Compose(tfl)



if __name__ == "__main__":
    dataframe = pd.read_csv('raw/train.csv', low_memory=False, squeeze=True)
    dataframe, label_mapping = create_folds(dataframe)
    train_dataset = DatasetRetriever('raw/', dataframe[dataframe['fold']!=0], transforms_train(224), label_mapping=label_mapping)
    validation_dataset = DatasetRetriever('raw/', dataframe[dataframe['fold']==0], transforms_validation(224), label_mapping=label_mapping)
    print(train_dataset[0]['image'].max(), train_dataset[0]['image'].min())
    print(validation_dataset[0]['image'].max(), validation_dataset[0]['image'].min())