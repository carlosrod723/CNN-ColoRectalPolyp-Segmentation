# Import necessary libraries and packages
import os
import torch
import yaml
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from source.network import SimpleUNetPP  
from albumentations import Compose, Normalize, Resize

# Function to load the image
def load_image(image_path):
    img= cv2.imread(image_path)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Function to preprocess the image
def preprocess_image(image, transform):
    augmented= transform(image=image)
    img= augmented['image']
    img= img.transpose(2,0,1)  
    img= img / 255.0  
    return torch.tensor(img, dtype= torch.float32).unsqueeze(0)  

# Function to postprocess the output mask
def postprocess_mask(mask, original_shape):
    mask= mask.squeeze(0).cpu().numpy()  
    mask= mask.squeeze(0)  
    mask[mask > 0.5]= 255
    mask[mask <= 0.5]= 0

    # Resize to the original size
    mask= cv2.resize(mask, (original_shape[1], original_shape[0]))  
    return mask

# Load configuration
with open('config.yaml') as f:
    config= yaml.safe_load(f)

# Load the trained model
def load_model(model_path, device):
    model= SimpleUNetPP(num_classes=1, input_channels= 3, deep_supervision= True)  
    model.load_state_dict(torch.load(model_path, map_location= device))
    model.to(device)
    model.eval()
    return model

# Define a function to predict the segmentation mask
def predict(image_path, model, device):

    # Load and preprocess the image
    img= load_image(image_path)
    original_shape= img.shape
    transform= Compose([Resize(256, 256), Normalize()])
    img_tensor= preprocess_image(img, transform).to(device)
    
    # Predict the mask
    with torch.no_grad():
        output= model(img_tensor)
        if isinstance(output, list): 
            output= output[-1]
        output= torch.sigmoid(output)
    
    mask= postprocess_mask(output, original_shape)
    return img, mask

# Main function to run the prediction
if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(description= 'Predict segmentation mask using trained model')
    parser.add_argument('--image', type= str, required= True, help= 'Path to the input image')
    parser.add_argument('--output', type= str, default= 'prediction.png', help= 'Path to save the output mask')
    args= parser.parse_args()

    # Set device
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model= load_model(config['model_path'], device)

    # Predict the mask
    input_image, predicted_mask= predict(args.image, model, device)

    # Save and display the results
    cv2.imwrite(args.output, predicted_mask)
    plt.figure(figsize= (9,5))
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(input_image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask, cmap= 'gray')
    plt.axis('off')
    plt.show()
