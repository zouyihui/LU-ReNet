import os
import glob
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torch.utils import data
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torch


input = torch.randn(1,530*500,100,1, requires_grad=True)
print(input)
print(input.shape)
print(input.size())

target = torch.empty(1,530*500,100,1).random_(2)
print("********")
print(target)
print(target.shape)


a = input.sum(0)/target.sum(0)
print("+++++++++++++")
print(a)
print(a.shape)