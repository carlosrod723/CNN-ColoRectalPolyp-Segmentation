# Import necessary libraries and packages
import numpy as np
import pandas as pd
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from source.utils import iou_score, AverageMeter
from sklearn.model_selection import train_test_split
from source.network import SimpleUNetPP
from source.dataset import DataSet
from src.data_augmentation import get_train_augmentations, get_val_augmentations

# Load configuration
with open('config.yaml') as f:
    config= yaml.safe_load(f)

extn= config['extn']
epochs= config['epochs']
log_path= config['log_path']
mask_path= config['mask_path']
image_path= config['image_path']
model_path= config['model_path']

# Create Log File
log= OrderedDict([
    ('epoch', []),
    ('loss', []),
    ('iou', []),
    ('val_loss', []),
    ('val_iou', []),
])

best_iou= 0
trigger= 0

# Split images into train and validation set
extn_= f'*{extn}'
img_ids= glob(os.path.join(image_path, extn_))
img_ids= [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
train_img_ids, val_img_ids= train_test_split(img_ids, test_size= 0.2)

# Get data augmentations
train_transform= get_train_augmentations()
val_transform= get_val_augmentations()

# Create train and validation dataset
train_dataset= DataSet(
    img_ids= train_img_ids,
    img_dir= image_path,
    mask_dir= mask_path,
    img_ext= extn,
    mask_ext= extn,
    transform= train_transform)

val_dataset= DataSet(
    img_ids= val_img_ids,
    img_dir= image_path,
    mask_dir =mask_path,
    img_ext= extn,
    mask_ext= extn,
    transform= val_transform)

# Create train and validation data loaders
train_loader= torch.utils.data.DataLoader(
    train_dataset,
    batch_size= 8,
    shuffle= True,
    drop_last= True)

val_loader= torch.utils.data.DataLoader(
    val_dataset,
    batch_size= 8,
    shuffle= False,
    drop_last= False)

# Initialize the model
model= SimpleUNetPP(num_classes=1, input_channels=3, deep_supervision=True)

# Define the combined Loss Function
def dice_loss(pred, target, smooth= 1.0):

    # Apply the sigmoid to the prediction
    pred= torch.sigmoid(pred)  
    intersection= (pred * target).sum()
    union= pred.sum() + target.sum()
    dice= (2. * intersection + smooth) / (union + smooth)
    return 1 - dice  

def combined_loss(pred, target, smooth= 1.0):
    pred= pred.squeeze(1)
    target= target.squeeze(1)
    bce= nn.BCEWithLogitsLoss()(pred, target)
    dice= dice_loss(pred, target, smooth)
    return bce + dice

params= filter(lambda p: p.requires_grad, model.parameters())

# Define Optimizer with a lower learning rate
optimizer= optim.Adam(params, lr= 3e-4, weight_decay= 1e-4)

# Define Learning Rate Scheduler
scheduler= optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Define the train function
def train(train_loader, model, criterion, optimizer, deep_sup= True):
    avg_meters= {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()
    pbar= tqdm(total=len(train_loader))
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for input, target, _ in train_loader:
        input= input.to(device)
        target= target.to(device)

        if deep_sup:
            outputs= model(input)
            loss= 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou= iou_score(outputs[-1], target)
        else:
            output= model(input)
            loss= criterion(output, target)
            iou= iou_score(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix= OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])

# Define the validate function
def validate(val_loader, model, criterion, deep_sup= True):
    avg_meters= {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.eval()
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        pbar= tqdm(total= len(val_loader))
        for input, target, _ in val_loader:
            input= input.to(device)
            target= target.to(device)

            if deep_sup:
                outputs= model(input)
                loss= 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou= iou_score(outputs[-1], target)
            else:
                output= model(input)
                loss= criterion(output, target)
                iou= iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])

# Run the train loop
for epoch in range(epochs):
    print(f'Epoch [{epoch} / {epochs}]')

    # Train for one epoch
    train_log= train(train_loader, model, combined_loss, optimizer, deep_sup=True)

    # Evaluate on validation set
    val_log= validate(val_loader, model, combined_loss, deep_sup=True)

    print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f' % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

    log['epoch'].append(epoch)
    log['loss'].append(train_log['loss'])
    log['iou'].append(train_log['iou'])
    log['val_loss'].append(val_log['loss'])
    log['val_iou'].append(val_log['iou'])

    pd.DataFrame(log).to_csv(log_path, index= False)

    trigger += 1

    if val_log['iou'] > best_iou:
        torch.save(model.state_dict(), model_path)
        best_iou = val_log['iou']
        print('=> saved best model')
        trigger= 0

    # Adjust learning rate based on validation loss
    scheduler.step(val_log['loss'])