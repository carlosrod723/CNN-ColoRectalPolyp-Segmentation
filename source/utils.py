# Import necessary libraries and packages
import torch

# Intersection over Union (IoU) Score Calculation
def iou_score(output, target, threshold= 0.5):
    smooth= 1e-5

    # Apply sigmoid to the output to bring it in range [0,1]
    if torch.is_tensor(output):
        output= torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target= target.data.cpu().numpy()

    # Apply threshold to obtain binary mask
    output_= output > threshold
    target_= target > threshold

    # Calculate intersection and union
    intersection= (output_ & target_).sum()
    union= (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

# Class to Track the Average of a Metric (like IoU)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val= 0
        self.avg= 0
        self.sum= 0
        self.count= 0

    def update(self, val, n= 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count