from test_coord_conv import AddCoordinates,CoordConv
import torch.nn as nn
from torch.nn import functional as F
import torch

im = torch.randn(1,3,530,500)
# c = nn.Conv2d(3,4,kernel_size=3,stride=1,padding=1)
# output=c(im)
# print(im.shape)
# print(output.shape)
# print("参数:")
# print(list(c.parameters()))
#
# coord_adder = AddCoordinates(True,False)
# coord_conv = CoordConv(3,5,3,with_r=True,usegpu=False)
# output=coord_adder(im)
# print(im.shape)
# print(output.shape)
# output2=coord_conv(im)
# print(output2.shape)
#
#
#
# print("参数:")
# print(list(c.parameters()))
###########
# class Net(nn.Module):
# #     def __init__(self):
# #         super(Net,self).__init__()
# #         self.conv1=CoordConv(3,5,3)
# #         self.conv2=CoordConv(3,5,3)
# #
# #     def forward(self,x):
# #         x =F.relu(self.conv1(x))
# #         x=F.relu(self.conv2(x))
# #
# #         return x
# #
# # net = Net()
# #
# # print("自定义网络:")
# # print(net)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = CoordConv(3,100,3,with_r=True,usegpu=False)
        self.conv2 = CoordConv(100,100,3,with_r=True,usegpu=False)
        self.conv3 = CoordConv(100,100,3,with_r=True,usegpu=False)
        self.conv4 = CoordConv(100,100,3,with_r=True,usegpu=False)
        self.conv5 = CoordConv(100,100,3,with_r=True,usegpu=False)
        self.conv6 = CoordConv(100,100,3,with_r=True,usegpu=False)
        self.conv6 = CoordConv(100,100,3,with_r=True,usegpu=False)

        self.fc1 = nn.Linear(100*3*3,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)


        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

net = Net()
print(net)


