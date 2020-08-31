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

# transform1 = transforms.Compose([
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor()
# ])
#
# class SegDataset(Dataset):
#     def __init__(self, list_path, img_root, height, width, number_of_instances, semantic_ann_npy, instances_ann_npy, transform=None):
#         super(SegDataset,self).__init__()
#         self.root = img_root
#         self.list_path = list_path
#         #self.mean = np.asarray(mean, np.float32)
#         self.height = height
#         self.width = width
#         self.semantic_ann_npy = semantic_ann_npy
#         self.instances_ann_npy = instances_ann_npy
#         self.transform = transform1
#         self.img_ids = [i_id.strip().split("\\")[-1].split(".")[0] for i_id in open(list_path)]
#         self.number_of_instance = [num_of_ins.strip().split(",") for num_of_ins in open(number_of_instances)]
#
#
#         self.toTensor = transform
#
#         self.files = []
#         for name in self.img_ids:
#             img_file = os.path.join(self.root,"%s_rgb.png" %name)
#             semantic_file = os.path.join(self.root, "%s_fg.png" %name)
#             instance_file = os.path.join(self.root, "%s_lable.png" %name)
#             semantic_npy = os.path.join(self.semantic_ann_npy, "%s.npy" %name)
#             instance_npy = os.path.join(self.instances_ann_npy, "%s.npy" %name)
#             for i in range(len(self.number_of_instance)):
#                 if self.number_of_instance[i][0] ==name:
#                     self.files.append({
#                         "img":img_file,
#                         "semantic_file":np.load(semantic_npy),
#                         "instance_file":np.load(instance_npy),
#                         "height":self.height,
#                         "width":self.width,
#                         "n_object":self.number_of_instance[i][1],
#                         "name":name
#             })
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, index):
#         datafiles = self.files
#
#         name = datafiles[index]["name"]
#         image = Image.open(datafiles[index]["img"]).convert('RGB')
#         if self.toTensor:
#             image = self.toTensor(image)
#             #return image
#         semantic_ann = datafiles[index]["semantic_file"]
#         instance_ann = datafiles[index]["instance_file"]
#         height = datafiles[index]["height"]
#         width = datafiles[index]["width"]
#         n_objects = torch.IntTensor(int(datafiles[index]["n_object"]))
#
#
#
#
#
#         return image,semantic_ann,instance_ann,n_objects
#
# if __name__ == '__main__':
#     img_root = 'F://1/instance-segmentation-pytorch-master/data/raw/CVPPP/CVPPP2017_LSC_training/training/A1/'
#     list_path = 'F://1/instance-segmentation-pytorch-master/data/metadata/CVPPP/train_list_file.txt'
#     number_of_instances = 'F://1/instance-segmentation-pytorch-master/data/metadata/CVPPP/number_of_instances.txt'
#     semantic_ann_npy = 'F://1/instance-segmentation-pytorch-master/data/processed/CVPPP/semantic-annotations/'
#     instances_ann_npy = 'F://1/instance-segmentation-pytorch-master/data/processed/CVPPP/instance-annotations/'
#
#
#     dst = SegDataset(list_path=list_path, img_root=img_root, height=530, width=500, number_of_instances=number_of_instances,semantic_ann_npy=semantic_ann_npy, instances_ann_npy=instances_ann_npy, transform = transform1 )
#     trainloader = data.DataLoader(dst,batch_size = 1)
#
#     for i,data in enumerate(trainloader):
#         a,b,c,d=data
#         print(c.shape)
#         print(d.shape)

##################################'原始dataset'

transform1 = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

class SegDataset(Dataset):
    def __init__(self, list_path, img_root, height, width, number_of_instances, semantic_ann_npy, instances_ann_npy, transform=None):
        super(SegDataset,self).__init__()
        self.root = img_root
        self.list_path = list_path
        #self.mean = np.asarray(mean, np.float32)
        self.height = height
        self.width = width
        self.semantic_ann_npy = semantic_ann_npy
        self.instances_ann_npy = instances_ann_npy
        self.transform = transform1
        self.img_ids = [i_id.strip().split("\\")[-1].split(".")[0] for i_id in open(list_path)]
        self.number_of_instance = [num_of_ins.strip().split(",") for num_of_ins in open(number_of_instances)]


        self.toTensor = transform

        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root,"%s_rgb.png" %name)
            semantic_file = os.path.join(self.root, "%s_fg.png" %name)
            instance_file = os.path.join(self.root, "%s_lable.png" %name)
            semantic_npy = os.path.join(self.semantic_ann_npy, "%s.npy" %name)
            instance_npy = os.path.join(self.instances_ann_npy, "%s.npy" %name)
            for i in range(len(self.number_of_instance)):
                if self.number_of_instance[i][0] ==name:
                    self.files.append({
                        "img":img_file,
                        "semantic_file":np.load(semantic_npy),
                        "instance_file":np.load(instance_npy),
                        "height":self.height,
                        "width":self.width,
                        "n_object":self.number_of_instance[i][1],
                        "name":name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files

        name = datafiles[index]["name"]
        image = Image.open(datafiles[index]["img"]).convert('RGB')
        if self.toTensor:
            image = self.toTensor(image)
            #return image
        semantic_ann = datafiles[index]["semantic_file"]
        instance_ann = datafiles[index]["instance_file"]
        height = datafiles[index]["height"]
        width = datafiles[index]["width"]
        n_objects = int(datafiles[index]["n_object"])


        return image,semantic_ann,instance_ann,n_objects

class AlignCollate(object):
    def __init__(self, n_classes, max_n_objects, image_height, image_width):

        self.n_classes = n_classes
        self.max_n_objects = max_n_objects
        self.image_height = image_height
        self.image_width = image_width

    def __preprocess(self, image, semantic_annotation, instance_annotation):

        # instance_annotation = instance_annotation.transpose(2, 0, 1)
        # n_objects = len(instance_annotation)
        # instance_annotation_resized = []
        #
        # for i in range(n_objects):
        #     instance_annotation_resized.append(instance_annotation)
        #
        # for i in range(self.max_n_objects-n_objects):
        #     zero = np.zeros((self.image_height,self.image_width),dtype=np.uint8)
        #     zero = Image.fromarray(zero)
        #     zero = np.array(zero)
        #     zero = np.expand_dims(zero,axis=0)
        #     instance_annotation_resized.append(zero.copy())
        #
        # instance_annotation_resized = np.vstack(instance_annotation_resized)
        # instance_annotation_resized = instance_annotation_resized.transpose(1, 2, 0)

        instance_annotation = instance_annotation.transpose(2,0,1)
        n_objects = len(instance_annotation)
        for i in range(self.max_n_objects-n_objects):
            zero = np.zeros((self.image_height,self.image_width),dtype=np.uint8)
            zero = np.expand_dims(zero,axis=0)
            instance_annotation = np.vstack((instance_annotation,zero))
        instance_annotation = instance_annotation.transpose(1,2,0)


        return (image, semantic_annotation,instance_annotation )#instance_annotation_resized

    def __call__(self, batch):
        images, semantic_annotations, instance_annotations, n_objects = zip(*batch)

        images = list(images)
        semantic_annotations = list(semantic_annotations)
        instance_annotations = list(instance_annotations)

        bs = len(images)
        for i in range(bs):
            img, semantic_annotation, instance_annotation = self.__preprocess(images[i],semantic_annotations[i],instance_annotations[i])

            images[i] = img
            semantic_annotations[i] = semantic_annotation
            instance_annotations[i] = instance_annotation

        images = torch.stack(images)

        instance_annotations = np.array(instance_annotations,dtype = 'int')

        semantic_annotations = np.array(semantic_annotations,dtype='int')
        semantic_annotations_one_hot = np.eye(self.n_classes,dtype='int')
        semantic_annotations_one_hot = semantic_annotations_one_hot[semantic_annotations.flatten()].reshape(semantic_annotations.shape[0], semantic_annotations.shape[1],semantic_annotations.shape[2], self.n_classes)

        instance_annotations = torch.LongTensor(instance_annotations)
        instance_annotations = instance_annotations.permute(0,3,1,2)

        semantic_annotations_one_hot = torch.LongTensor(semantic_annotations_one_hot)
        semantic_annotations_one_hot = semantic_annotations_one_hot.permute(0,3,1,2)

        n_objects = torch.IntTensor(n_objects)

        return (images,semantic_annotations_one_hot,instance_annotations,n_objects)

if __name__ == '__main__':
    img_root = 'F://1/instance-segmentation-pytorch-master/data/raw/CVPPP/CVPPP2017_LSC_training/training/A1/'
    list_path = 'F://1/instance-segmentation-pytorch-master/data/metadata/CVPPP/train_list_file.txt'
    number_of_instances = 'F://1/instance-segmentation-pytorch-master/data/metadata/CVPPP/number_of_instances.txt'
    semantic_ann_npy = 'F://1/instance-segmentation-pytorch-master/data/processed/CVPPP/semantic-annotations/'
    instances_ann_npy = 'F://1/instance-segmentation-pytorch-master/data/processed/CVPPP/instance-annotations/'

    dst = SegDataset(list_path=list_path, img_root=img_root, height=530, width=500,
                     number_of_instances=number_of_instances, semantic_ann_npy=semantic_ann_npy,
                     instances_ann_npy=instances_ann_npy, transform=transform1)
    #iimage, semantic_annotation,instance_annotation,n_objects = dst[2]
    #trainloader = data.DataLoader(dst,batch_size=1)
    # for i ,data in enumerate(trainloader):
    #     a,b,c,d = data
    #
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    # print(d.shape)
    #print(np.unique(instance_annotation))
    # print('*****************')
    ac = AlignCollate(9,100,530,500)

    trainloader = torch.utils.data.DataLoader(dst,batch_size=1,collate_fn=ac)
    #loader = iter(loader)
    for i,data in enumerate(trainloader):
        a,b,c,d = data
        print(a.shape)
        print(b.shape)
        print(c.shape)
        print(d.shape)
    #images,semantic_annotations,instance_annotations,n_objects=loader.next()

    # print(images.size())
    # print(semantic_annotations.size())
    # print(instance_annotations.size())
    # print(n_objects.size())
    #print(n_objects)
    #print(semantic_annotations)
    #print(np.unique(semantic_annotation))