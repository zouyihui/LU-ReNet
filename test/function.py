import torch
import numpy as np
import torchsnooper
from sklearn.metrics import confusion_matrix

#n_objects处理后转换为int类型
def n_objects_prediction(n_object, max_n_objects):
    n_objects_predictions = n_object * max_n_objects
    n_objects_predictions = torch.round(n_objects_predictions).int()
    return n_objects_predictions

#calc_dic
def calc_dic(n_objects_gt,n_objects_pred):
    return np.abs(n_objects_gt-n_objects_pred)


def sem_seg_prediction(sem_seg_prediction):
    sem_seg_predictions = torch.nn.functional.softmax(sem_seg_prediction, dim=1)
    return sem_seg_predictions

#@torchsnooper.snoop()
#calc_dic(gt_seg, pred_seg)
def calc_dice(gt_seg, pred_seg):

    nom = 2.0*torch.sum(gt_seg * pred_seg) #np.sum
    denom = torch.sum(gt_seg) + torch.sum(pred_seg)

    dice = float(nom)/ float(denom)
    return dice

#calc_bd
def calc_bd(ins_seg_gt, ins_seg_pred):
    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred.detach().numpy())).difference([0]))
    best_dices = []

    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        dices = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    best_dice = np.mean(best_dices)
    return best_dice

def calc_sbd(ins_seg_gt, ins_seg_pred):

    _dice1 = calc_bd(ins_seg_gt, ins_seg_pred)
    _dice2 = calc_bd(ins_seg_pred, ins_seg_gt)
    return min(_dice1, _dice2)

#计算混淆矩阵
def con_matrix(semantic_annotations, sem_pred):

    sem_pred = sem_pred.detach().numpy()
    semantic_annotations = semantic_annotations.squeeze()
    sem_pred = sem_pred.squeeze()

    sem_pred=np.argmax(sem_pred, axis=0)

    semantic_annotations = semantic_annotations.flatten()
    sem_pred = sem_pred.flatten()

    c_matrix = confusion_matrix(semantic_annotations,sem_pred)

    return c_matrix