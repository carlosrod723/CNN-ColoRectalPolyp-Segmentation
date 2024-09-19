# Import necessary libraries and packages
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Create a class for defining a custom dataset using images and masks for image segmentation
class DataSet(Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext= '.png', mask_ext= '.png', transform= None):
        '''
        Args:
            img_ids (list): List of image IDs to load.
            img_dir (str): Directory path where images are stored.
            mask_dir (str): Directory path where masks are stored.
            img_ext (str): Extension of image files.
            mask_ext (str): Extension of mask files.
            transform (callable, optional): Optional transform to be applied on a sample.
        '''

        self.img_ids= img_ids
        self.img_dir= img_dir
        self.mask_dir= mask_dir
        self.img_ext= img_ext
        self.mask_ext= mask_ext
        self.transform= transform

    def __len__(self):
        '''Returns the total number of samples in the dataset.'''

        return len(self.img_ids)
    
    def __getitem__(self, index):
        '''
        Load and return a sample from the dataset at the given index.
        
        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, mask, {'img_id': img_id}) where image is the processed image tensor, mask is the processed 
            mask tensor, and img_id is the ID of the image.
        '''

        img_id= self.img_ids[index]

        # Load image
        img= cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        
        # Load mask
        mask= cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]
        mask= np.dstack([mask])

        # Apply transformations
        if self.transform is not None:
            augmented= self.transform(image= img, mask= mask)
            img= augmented['image']
            mask= augmented['mask']

        # Normalize and convert to tensor format
        img= img.astype('float32') / 255
        img= img.transpose(2, 0, 1)
        mask= mask.astype('float32') / 255
        mask= mask.transpose(2, 0, 1)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype= torch.float32), {'img_id': img_id}
