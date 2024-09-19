# Import necessary libraries and packages
import os
import numpy as np
import cv2

# Set the path to your dataset
image_path= 'images/Original'
img_ext= '.png'

# Initialize variables to calculate mean and standard deviation
sum_mean= 0
sum_std= 0
num_pixels= 0

# Get all image file names
img_ids= [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith(img_ext)]

for img_id in img_ids:

    # Load image
    img= cv2.imread(os.path.join(image_path, img_id + img_ext), cv2.IMREAD_GRAYSCALE)

    # Normalize to range [0,1]
    img= img / 255.0
    
    # Calculate mean and standard deviation for the current image
    img_mean= np.mean(img)
    img_std= np.std(img)
    
    # Accumulate the sum
    sum_mean += img_mean
    sum_std += img_std
    num_pixels += 1

# Calculate the overall mean and standard deviation
dataset_mean= sum_mean / num_pixels
dataset_std= sum_std / num_pixels

print(f'Dataset Mean: {dataset_mean:.4f}, Dataset Std: {dataset_std:.4f}')
