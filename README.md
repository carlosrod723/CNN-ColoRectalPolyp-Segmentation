# Computer Vision CNN-Based Colorectal Polyp Segmentation

## Overview

A computer vision-based deep learning solution designed for automatic segmentation of colorectal polyps from colonoscopy images, leveraging convolutional neural networks (CNNs). The primary objective is to assist medical professionals in detecting and analyzing polyps, which is crucial in the early prevention of colorectal cancer.

The model architecture used is a variant of U-Net++, specifically adapted for this medical imaging task. Deep supervision is incorporated to ensure the learning of meaningful representations across multiple layers. Various preprocessing and postprocessing techniques are employed to ensure robustness and accuracy in polyp segmentation. This approach demonstrates the effectiveness of applying computer vision techniques to the healthcare domain, enhancing diagnostic support and decision-making for physicians.

## Aim

The aim of this work is to provide a reliable and accurate deep learning model capable of segmenting polyps from medical imaging data. Through efficient model design, data augmentation, and custom loss functions, the solution is built to handle the complexity and variability present in medical imaging datasets.

## Data

The dataset consists of colonoscopy images with their corresponding binary masks, which highlight the areas of polyps. The data is stored in two directories:

- **Original**: Contains the raw colonoscopy images.
- **Ground_Truth**: Contains binary masks, where polyp regions are marked.

Each image and its corresponding mask are in `.png` format. The model processes images at a resolution of `384x288`.

## Contents

The repository is structured as follows:

- **images/**
  - **Original/**: Contains the original colonoscopy images.
  - **Ground_Truth/**: Contains the binary masks that mark polyp regions.
  
- **models/**
  - **model.pth**: The trained model file saved after the best validation performance.
  
- **logs/**
  - **logs.csv**: Logs of the training process, including loss and IoU metrics for each epoch.
  
- **source/**
  - **dataset.py**: Defines the custom dataset class for loading, preprocessing, and augmenting images and masks.
  - **network.py**: Contains the implementation of the U-Net++ architecture, including VGG-like blocks and deep supervision.
  - **utils.py**: Utility functions, including IoU calculation and average metric tracking during training.
  
- **src/**
  - **data_augmentation.py**: Handles data augmentation operations, such as rotation, flipping, and normalization.
  - **data_inspection.py**: Contains tools for dataset inspection and statistics calculation.
  
- **config.yaml**: Configuration file defining parameters like file paths, learning rate, batch size, etc.
  
- **train.py**: Script to train the U-Net++ model, using the dataset and augmentation techniques.
  
- **predict.py**: Script for running inference on new images using the trained model to generate polyp segmentation masks.
  
- **requirements.txt**: Contains the Python dependencies needed to run the code.
  
- **README.md**: This file.

## Key Components

### Model Architecture

The model is a modified version of U-Net++, designed to address the challenges of medical image segmentation. U-Net++ consists of nested U-Net architectures that enhance feature propagation and reuse, resulting in more accurate segmentation outputs. The model incorporates the following key features:

- **VGG-style blocks** for convolutional operations, which help in extracting complex features from the input images.
- **Deep supervision**, where intermediate layers contribute to the overall loss, ensuring gradient flow throughout the network and helping prevent vanishing gradients.

### Loss Function

The combined loss function uses a mix of Dice Loss and Binary Cross-Entropy (BCE) Loss. This hybrid approach helps in balancing pixel-wise classification accuracy (via BCE) and overlap between predicted and ground truth masks (via Dice Loss). Dice Loss is particularly effective in handling the imbalance between the polyp and non-polyp pixels, which is a common issue in medical image segmentation.

### Data Augmentation

Data augmentation is critical in addressing the small dataset size and variability in polyp appearance. The following transformations are applied:
- **Rotations and flips** to simulate different viewing angles of the same region.
- **Random brightness and contrast adjustments** to account for variations in lighting conditions during imaging.
- **Resizing** and **normalization** to ensure the data is uniform before feeding it to the model.

These augmentations improve the model’s generalization capabilities, enabling it to perform well on unseen data.

### Data Loading and Preprocessing

A custom dataset class is responsible for loading, augmenting, and preprocessing the images and masks. This class ensures that:
- Images are loaded and resized to the appropriate dimensions.
- Images are normalized, and masks are binarized, converting grayscale values into binary values (0 and 1).
- Data is transformed into tensors, making it ready for model training.

### Training Process

During the training process, the model learns to predict polyp masks by minimizing the combined loss function. The training is conducted over a set number of epochs, during which the model's weights are adjusted using backpropagation and the Adam optimizer. A learning rate scheduler is employed to reduce the learning rate dynamically when the validation loss plateaus, helping the model converge more effectively.

Each epoch involves:
- Training on batches of images and masks, with the model updating its weights based on the combined loss.
- Validation, where the model's performance is evaluated on a separate set of data to monitor overfitting and generalization.
- Metrics such as Intersection over Union (IoU) are calculated to evaluate the overlap between the predicted mask and the actual ground truth mask. This metric provides insight into the segmentation accuracy.

### Model Evaluation

Once the training process completes, the model is evaluated on validation data using key metrics:
- **Loss**: Quantifies the overall error in the model’s predictions.
- **IoU (Intersection over Union)**: Measures the overlap between predicted and actual masks. Higher IoU scores indicate better segmentation performance.

Currently, the training shows promising results, although the IoU on the validation set suggests that more training epochs are necessary for the model to fully converge. The initial results are encouraging, and running additional epochs is likely to improve performance significantly.

### Inference and Prediction

The trained model is used for inference on unseen images. The prediction process involves:
- Preprocessing the input image by resizing and normalizing it.
- Feeding the image into the trained U-Net++ model.
- Generating a binary mask that highlights the polyp regions in the image.

The predicted mask is resized to match the original image dimensions for accurate visualization. These predictions can be further used in clinical settings for real-time polyp detection and analysis.

## Conclusion

The work demonstrates a sophisticated yet scalable approach to polyp segmentation using deep learning. While the model architecture and techniques applied have shown potential, further refinement through extended training and model tuning will enhance the segmentation accuracy. The results indicate that the model is on the right track, and with additional epochs, it will converge and perform effectively on both training and unseen data.

This repository provides a robust starting point for applying deep learning to medical image segmentation, specifically in detecting colorectal polyps. The combination of U-Net++, deep supervision, data augmentation, and custom loss functions make this approach highly adaptable to various medical imaging challenges.
