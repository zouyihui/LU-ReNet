import torch.nn as nn
import torchvision.models as models
from coor_conv import CoordConvNet
from coor_conv import CoordConv
import torch
from renet import ReNet
from instance_counter import InstanceCounter
from coor_vgg16 import VGG16
from Xception import Xception


class ReSeg(nn.Module):
    """
    * VGG16 with skip Connections as base network
    * Two ReNet layers
    * Two transposed convolutional layers for upsampling
    * Three heads for semantic segmentation, instance segmentation and
        instance counting.

        n_classes (int): Number of semantic classes
        use_instance_seg (bool, optional): If `False`, does not perform
            instance segmentation. Default: `True`
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`


            Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output:
            - Semantic Seg: `(N, N_{class}, H_{in}, W_{in})`
            - Instance Seg: `(N, 32, H_{in}, W_{in})`
            - Instance Cnt: `(N, 1)`"""
    def __init__(self, n_classes, use_instance_seg=True, pretrained=False, use_coordinates=True,usegpu=False):
        super(ReSeg, self).__init__()

        self.n_classes = n_classes
        self.use_instnce_seg = use_instance_seg


        #Encoder
        #BaseCNN
        self.cnn = Xception(pretrained=False, use_coordinates=True, usegpu=usegpu)

        #ReNets
        self.renet1 = ReNet(512, 256, use_coordinates=use_coordinates,usegpu=usegpu)

        self.renet2 = ReNet(256*2, 256,use_coordinates=use_coordinates,usegpu=usegpu)

        # Decoder
        self.upsampling1 = nn.ConvTranspose2d( 1024, 128,kernel_size=(2,2),stride=(2,2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(128)
        self.upsampling2 = nn.ConvTranspose2d(256, 64, kernel_size=(2,2), stride=(2,2))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)

        #semantic segmentation
        self.sem_seg_out_put = nn.Conv2d(128 , self.n_classes,kernel_size=(1,1),stride=(1,1)) #100+64

        #instance segmentation
        if self.use_instnce_seg:
            self.ins_seg_output = nn.Conv2d(128 , 20, kernel_size=(1,1),stride=(1,1))
        #
        #Instance Counting
        #self.ins_cls_cnn = InstanceCounter(64*2 , use_coordinates=True,usegpu=False)

    def forward(self, x):
        #Encoder
        #BaseCNN
        zero_skip, first_skip, second_skip, x_enc = self.cnn(x)  #first_skip , second_skip,

        #ReNets
        x_enc = self.renet1(x_enc)
        x_enc = self.renet2(x_enc)

        #Decoder
        # x_dec = self.relu1(self.upsampling1(x_enc))
        # x_dec = torch.cat([x_dec, second_skip], dim=1)
        # x_dec = self.relu2(self.upsampling2(x_dec))
        # x_dec = torch.cat([x_dec,first_skip],dim=1)
        #x_dec = x_enc + second_skip

        x_dec = torch.cat([x_enc,second_skip], dim=1)
        x_dec = self.relu1(self.upsampling1(x_dec))   #1*100*250*250
        x_dec = self.bn1(x_dec)
        #x_dec = x_dec+first_skip
        x_dec = torch.cat([x_dec,first_skip],dim=1)
        x_dec = self.relu2(self.upsampling2(x_dec))
        x_dec = self.bn2(x_dec)
        x_dec = torch.cat([x_dec,zero_skip],dim=1)


        #Semantic Segmentation
        sem_seg_out = self.sem_seg_out_put(x_dec)

        #Instance Segmentation
        if self.use_instnce_seg:
            ins_seg_out = self.ins_seg_output(x_dec)
        # else:
        #     ins_seg_out = None

        #Instance Counting
        ins_cls_out = self.ins_cls_cnn(x_dec)

        return sem_seg_out,ins_cls_out,ins_seg_out
        #, ins_seg_out, ins_cls_out