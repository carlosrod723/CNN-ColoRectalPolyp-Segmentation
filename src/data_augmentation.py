# Import necessary libraries and packages
from albumentations import HorizontalFlip, RandomRotate90, Normalize, Resize, Compose

#  Define the data augmentation pipeline for training
def get_train_augmentations():
    return Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Resize(256, 256),
        Normalize(mean=[0.5], std=[0.5]),  
    ])

# Define the data augmentation pipeline for validation
def get_val_augmentations():
    return Compose([
        Resize(256, 256),
        Normalize(mean=[0.5], std=[0.5]),  
    ])
