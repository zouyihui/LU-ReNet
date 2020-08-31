import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
import torch.nn as nn
import torchvision.models as models
from coor_conv import CoordConvNet
from coor_conv import CoordConv
import torch
from renet import ReNet
from instance_counter import InstanceCounter
import torchsnooper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skiprelu = nn.ReLU(out_filters)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skiprelu(skip)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Acoor_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Acoor_Conv,self).__init__()
        self.conv = nn.Sequential(
            CoordConv(in_ch, out_ch, 3,padding=1, with_r=True, usegpu=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CoordConv(out_ch, out_ch, 3,padding=1, with_r=True, usegpu=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,input):
        return self.conv(input)

class Xception(nn.Module):
    def __init__(self, pretrained=False, use_coordinates=True,usegpu=False):
        super(Xception, self).__init__()
        self.use_coordinates = use_coordinates
        if self.use_coordinates:
            self.conv1= Acoor_Conv(3,64) #500,500
            self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)  #250
            self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)     #125
            self.conv2 = Acoor_Conv(256,512)
            self.conv3 = CoordConv(512,512,3,padding=1, with_r=True, usegpu=False)
            self.conv4 = nn.ReLU(inplace=True)
            self.relu4 = nn.BatchNorm2d(512)
    def forward(self, x):

        c1 = self.conv1(x)
        b1 = self.block1(c1)
        b2 = self.block2(b1)
        c2 = self.conv2(b2)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.relu4(c4)


        return c1,b1 ,c2,c5

if __name__ == '__main__':
    from torch.autograd import Variable
    import torch
    import time

    def test():
        batch_size,image_size = 1, 128
        xception = Xception(use_coordinates=True,usegpu=False)
        input = Variable(torch.rand(batch_size, 3, image_size, image_size))
        xception = xception.to(device)
        input = input.to(device)
        output = xception(input)
        print(output[0].shape,output[1].shape,output[2].shape)
    test()