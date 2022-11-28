import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from content_net_model.model_of_contentNet import ContentNet
import os
'''
Loss functions and qualitative evaluation functions used in training and validating.
'''

########################################################
# Loss Functions For SDNet
########################################################


def gradient_loss(s):
    '''
    Smooth Loss : Calculate the smoothing loss of the registration field.
    :param s: Registration Field
    :return: loss value
    '''
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
    dy = dy * dy
    dx = dx * dx
    d = torch.mean(dx) + torch.mean(dy)
    whole_grid_loss = d/2.0
    return whole_grid_loss


class ContentLoss(nn.Module):
    '''
    Content Loss
    '''
    def __init__(self):
        super().__init__()
        self.model = ContentNet(embedding1_length=512, embedding2_length=256)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'content_net_model/out/model.pth')
        self.__load_model(model_path)
        self.eval().requires_grad_(False)

    def __load_model(self, path):
        if os.path.exists(path):
            state = torch.load(path)
            self.model.load_state_dict(state['net'])
            print('Successful to load model parameters of ContentNet')
        else:
            print('Failed to load model parameters of ContentNet')

    def forward(self, input_, label_):
        '''
        Content Loss : Calculate the content loss between transformed stroke and label stroke.
        :param input_: transformed stroke, shape is (Batch, 1, 256, 256)
        :param label_: label stroke, shape is (Batch, 1, 256, 256)
        :return: content loss value
        '''
        embedding_out, recon_out = self.model(input_)
        embedding_label, recon_label = self.model(label_)
        embedding_out = F.normalize(embedding_out, p=2, dim=1)
        embedding_label = F.normalize(embedding_label, p=2, dim=1)
        content_loss = torch.sum((embedding_out - embedding_label) ** 2, dim=1)
        content_loss = torch.mean(content_loss)
        return content_loss



















########################################################
# val Functions For SDNet
########################################################


def get_centroid_box_qualitative_result(kaiti, style):
    '''
    calculate mDis and mBIou
    @param kaiti: (N,256,256), numpy
    @param style: (N,256,256), numpy
    @return:
    '''
    def centroid_box(img):
        point = np.where(img > 0.5)
        center = np.array([np.mean(point[1]), np.mean(point[0])])
        box = np.array([np.min(point[1]), np.max(point[1]),np.min(point[0]), np.max(point[0])]) # xl,xr,yt.yb
        return center, box

    mean_distance = []
    mean_box_iou = []

    for i in range(kaiti.shape[0]):
        kaiti_center, kaiti_box = centroid_box(kaiti[i])
        style_center, style_box = centroid_box(style[i])
        distance = np.sqrt(np.sum((kaiti_center-style_center)**2))
        or_box = (max(kaiti_box[1], style_box[1]) - min(kaiti_box[0], style_box[0]))* \
                 (max(kaiti_box[3], style_box[3]) - min(kaiti_box[2], style_box[2]))
        and_box = (min(kaiti_box[1], style_box[1]) - max(kaiti_box[0], style_box[0])) * \
                 (min(kaiti_box[3], style_box[3]) - max(kaiti_box[2], style_box[2]))
        iou = and_box/or_box
        mean_distance.append(distance)
        mean_box_iou.append(iou)
    mDis = np.mean(np.array(mean_distance))
    mBIou = np.mean(np.array(mean_box_iou))
    return mDis, mBIou

########################################################
# val Functions For SegNet
########################################################


def get_mean_IOU(out, label):
    '''
    Common semantic segmentation evaluation methods for SegNet
    '''
    out_ = out > 0.5
    label_ = label > 0.5
    orp = torch.logical_or(out_, label_)
    andp = torch.logical_and(out_, label_)
    and_sum = torch.sum(andp, dim=[2, 3])
    or_sum = torch.sum(orp, dim=[2, 3])
    iou_w = or_sum > 1
    t = torch.true_divide(and_sum+1, or_sum+1)*iou_w
    ts = torch.mean(t, dim=1)
    ts = ts*7/torch.sum(iou_w, dim=1)
    return torch.mean(ts)


########################################################
# val Functions For ExtractNet
########################################################


def get_iou_with_matching(out, label):
    '''
    calculate mIOUm
    '''
    ious = []
    for index, each in enumerate(out):
        iou = np.sum((each + label) > 1.5) / (np.sum((each + label) > 0.5) + 0.00001)
        ious.append(iou)
    return np.mean(np.array(ious))


def get_iou_without_matching(out, label):
    '''
    calculate mIOUum
    '''
    ious = []
    for index, each in enumerate(out):
        union = [np.sum((each+x)>1.5) for x in label]
        max_index = np.argmax(np.array(union))
        label_m = label[max_index]
        iou = np.sum((each + label_m) > 1.5) / (np.sum((each + label_m) > 0.5) + 0.00001)
        ious.append(iou)
    return np.mean(np.array(ious))