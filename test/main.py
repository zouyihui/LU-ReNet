import torch
from torchvision.transforms import transforms as T
import argparse
import coor_vgg16
from torch import optim
from dataset import SegDataset
from torch.utils.data import DataLoader
from coor_vgg16 import VGG16
from dataset import AlignCollate
from discriminative import DiscriminativeLoss
from dice import DiceLoss,DiceCoefficient,dice_coefficient
from reseg import ReSeg
from torch.autograd import Variable
from evaluate import calc_dice
from function import n_objects_prediction,sem_seg_prediction,calc_dic,calc_dice,calc_bd,calc_sbd,con_matrix



#device = torch.decive('cuda' if torch.cuda.is_available() else 'cpu')

x_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

y_transform = T.ToTensor()

def train_model(model,criterion1,criterion2,optimizer,dataload,num_epochs=100):
    for epoch in range(num_epochs):
        import time
        start = time.time()
        print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        epoch_dic = 0
        step = 0
        for x, y, instance_annotations,n_objects,semantic_annotations in dataload:
            optimizer.zero_grad()  # 每次minibatch都要讲梯度清零
            inputs = x
            seg_labels = y
            sem_seg_out,ins_seg_out = model(inputs)  # 前向传播
            sem_seg_out,n_objects_pred = sem_seg_prediction(sem_seg_out)
            n_objects_dic = calc_dic(n_objects,n_objects_pred)
            seg_loss = criterion1(sem_seg_out, seg_labels)

            seg_loss.backward()
            optimizer.step()
            epoch_loss += seg_loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, seg_loss.item()))
            dice_loss_1 = DiceLoss(weight=None)
            dice = dice_loss_1(sem_seg_out, seg_labels)

    torch.save(model.state_dict(),"Xception+coor_a1_weights_%d.pth" % epoch)  #返回模型的所有内容
    return model


img_root = ''
list_path = ''
number_of_instances = ''
semantic_ann_npy = ''
instances_ann_npy = ''

def train():
    model = ReSeg( 2,pretrained=False, use_coordinates=True, usegpu=False)
    batch_size = 2
    criterion1 = DiceLoss()
    criterion2 = DiscriminativeLoss(0.5,1.5,2)
    optimizer = optim.Adam(model.parameters())
    dst = SegDataset(list_path=list_path, img_root=img_root, height=448, width=448,
                     number_of_instances=number_of_instances, semantic_ann_npy=semantic_ann_npy,
                     instances_ann_npy=instances_ann_npy, transform=x_transform)
    ac = AlignCollate(2, 100, 448, 448)
    trainloader = torch.utils.data.DataLoader(dst, batch_size=1, collate_fn=ac)
    train_model(model, criterion1,criterion2, optimizer, trainloader)



if __name__ == '__main__':
    train()