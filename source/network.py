# Import necessary libraries and packages
import torch
from torch import nn

# Create a SimpleUNet++ class for image segmentation
class SimpleUNetPP(nn.Module):
    def __init__(self, num_classes, input_channels= 3, deep_supervision= False):
        super(SimpleUNetPP, self).__init__()

        nb_filter= [32,64,128]

        self.deep_supervision= deep_supervision

        self.pool= nn.MaxPool2d(2,2)
        self.up= nn.Upsample(scale_factor= 2, mode= 'bilinear', align_corners= True)

        self.conv0_0= self.conv_block(input_channels, nb_filter[0])
        self.conv1_0= self.conv_block(nb_filter[0], nb_filter[1])
        self.conv2_0= self.conv_block(nb_filter[1], nb_filter[2])

        self.conv0_1= self.conv_block(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1= self.conv_block(nb_filter[1] + nb_filter[2], nb_filter[1])

        self.conv0_2= self.conv_block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1= nn.Conv2d(nb_filter[0], num_classes, kernel_size= 1)
            self.final2= nn.Conv2d(nb_filter[0], num_classes, kernel_size= 1)
            self.final3= nn.Conv2d(nb_filter[0], num_classes, kernel_size= 1)
        else:
            self.final= nn.Conv2d(nb_filter[0], num_classes, kernel_size= 1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, kernel_size= 3, padding= 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True)
        )

    def forward(self, input):
        x0_0= self.conv0_0(input)
        x1_0= self.conv1_0(self.pool(x0_0))
        x0_1= self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0= self.conv2_0(self.pool(x1_0))
        x1_1= self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2= self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        if self.deep_supervision:
            output1= self.final1(x0_1)
            output2= self.final2(x0_2)
            return [output1, output2]
        else:
            output= self.final(x0_2)
            return output