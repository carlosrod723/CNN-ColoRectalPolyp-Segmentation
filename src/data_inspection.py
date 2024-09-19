# Import necessary libraries and packages
import os
import cv2
import yaml
import matplotlib.pyplot as plt

# Load configuration
with open('config.yaml') as f:
    config= yaml.safe_load(f)

# Paths for config
image_path= config['image_path']
mask_path= config['mask_path']

# Load and visualize a few images and masks
def visualize_data():
    image_files= os.listdir(image_path)[:5]
    for image_file in image_files:

        # Load image
        img= cv2.imread(os.path.join(image_path, image_file))
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load corresponding mask
        mask_file= image_file.replace('.png', config['extn'])
        mask= cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)

        # Visualize image and mask
        plt.figure(figsize= (9,4))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title('Image')
        plt.subplot(1,2,2)
        plt.imshow(mask, cmap= 'gray')
        plt.title('Mask')
        plt.show()

if __name__ == '__main__':
    visualize_data()