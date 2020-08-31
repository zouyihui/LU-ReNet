import torch.nn as nn
import torchvision.models as models
from coor_conv import CoordConvNet
from coor_conv import CoordConv
import torch
from renet import ReNet
from instance_counter import InstanceCounter
import torchsnooper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Acoor_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Acoor_Conv,self).__init__()
        self.conv = nn.Sequential(
            CoordConv(in_ch, out_ch, 3,padding=1, with_r=True, usegpu=False),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CoordConv(out_ch, out_ch, 3,padding=1, with_r=True, usegpu=False),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,input):
        return self.conv(input)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch , out_ch,3,padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)


class VGG16(nn.Module):

    def __init__(self, pretrained=True, use_coordinates=False, usegpu=False):   #, n_layers, return_intermediate_outputs=False
        super(VGG16, self).__init__()


        self.use_coordinates = use_coordinates
        # self.return_intermediate_outputs = return_intermediate_outputs
        #
        # self.cnn = models.__dict__['vgg16'](pretrained=pretrained)
        # self.cnn = nn.Sequential(*list(self.cnn.children())[0])
        # self.cnn = nn.Sequential(*list(self.cnn.children())[: n_layers])


        if self.use_coordinates:
            self.conv1 = Acoor_Conv(3, 64)   #500,500
            self.pool1 = nn.MaxPool2d(2,stride=2)  #250
            self.conv2 = Acoor_Conv(64, 128)
            self.pool2 = nn.MaxPool2d(2,stride=2)         #125
            self.conv3 = Acoor_Conv(128, 256)
            self.conv4 = CoordConv(256, 256, 3,padding=1, with_r=True, usegpu=False)
            self.conv5 = nn.ReLU(inplace=True)
            # self.pool3 = nn.MaxPool2d(2)
            # self.conv4 = Acoor_Conv(256,512)
            # self.conv5 = nn.Conv2d(512,256,1)
            # self.conv6 = nn.Conv2d(256,128,1)  #in_ch,out_ch,kernel_size
            # self.conv7 = nn.Conv2d(128,64,1)
            # self.conv8 = nn.Conv2d(64,3,1)
        else:
            self.conv1 = DoubleConv(3,64)
            self.pool1 = nn.MaxPool2d(2,stride=2)
            self.conv2 = DoubleConv(64,128)
            self.pool2 = nn.MaxPool2d(2, stride=2)
            self.conv3 = DoubleConv(128, 256)
            self.conv4 = nn.Conv2d(256,256,3,padding=1)
            self.conv5 = nn.ReLU(inplace=True)


    # def __get_outputs(self, x):
    #     if self.use_coordinates:
    #         return self.cnn(x)
    #
    #     outputs = []
    #     for i, layer in enumerate(self.cnn.children()):
    #         x = layer(x)
    #         outputs.append(x)

    #    return outputs


    def forward(self, x):
        # outputs = self.__get_outputs(x)
        #
        # if self.return_intermediate_outputs:
        #     return outputs
        #
        # return outputs[-1]

        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.conv4(c3)
        c4 = self.conv5(p3)
        # c5 = self.conv5(c4)
        # c6 = self.conv6(c5)
        # c7 = self.conv7(c6)
        # c8 = self.conv8(c7)

        #out = nn.Sigmoid()(c4)
        return p1,c3,c4



if __name__ == '__main__':
    from torch.autograd import Variable
    import torch
    import time

    def test():
        batch_size,image_size = 1, 128
        vgg16 = VGG16(False,use_coordinates=True,usegpu=False)
        input = Variable(torch.rand(batch_size, 3, image_size,image_size))
        vgg16 = vgg16.to(device)
        input = input.to(device)
        output = vgg16(input)
        print(output[0].shape,output[1].shape,output[2].shape)
        #print(output)
    # def test(use_coordinates, usegpu, skip):
    #
    #         batch_size, image_size = 1, 128
    #
    #         if skip:
    #             vgg16 = SkipVGG16(False, use_coordinates, usegpu)
    #         else:
    #             vgg16 = VGG16( False, use_coordinates, usegpu) #3, False
    #
    #         input = Variable(torch.rand(batch_size, 3, image_size,image_size))
    #
    #         if usegpu:
    #             vgg16 = vgg16.to(device)
    #             input = input.to(device)
    #
    #         print ('\nModel :\n\n', vgg16)
    #
    #         output = vgg16(input)
    #         print(output.shape)
    print('\n### CPU')
    test()
        # if isinstance(output, list):
        #     print ('\n* N outputs : ', len(output))
        #     for o in output:
        #         print ('** Output shape : ', o.size())
        # else:
        #     print ('\n** Output Shape : ', output.size())

    # print ('\n### COORDS + GPU + SKIP')
    # test(True, False, True)
    # print ('\n### COORDS + GPU')
    # test(True, False, False)
    # print ('\n### COORDS + CPU + SKIP')
    # test(True, False, True)
    # print ('\n### COORDS + CPU')
    # test(True, False, False)
    # print ('\n### GPU + SKIP')
    # test(False, False, True)
    # print ('\n### GPU')
    # test(False, False, False)
    # print ('\n### CPU + SKIP')
    # test(True, False, True)

