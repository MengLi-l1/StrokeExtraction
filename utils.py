import torch.nn as nn
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
import torchvision.utils as vutils
import colorsys
'''
 Some common functions used in training
'''

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


seg_colors = random_colors(24)


def apply_stroke(image, mask, color):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask[:, :] > 0.5, color[c] , image[:, :, c])
    return image


def apply_stroke_t(image, mask, color):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask[:, :] > 0.5, color[c]*0.6 + 0.4*image[:, :, c], image[:, :, c])
    return image


def save_picture(*args, title_list, path, nrow=8):
    '''
    save data as picture during training and testing
    '''
    var_list = []
    for input_data in args:
        if input_data.size(1) != 3:
            input_ = torch.sum(input_data, dim=1, keepdim=True)
        else:
            input_ = input_data
        input_[input_ > 1] = 1
        input_image = vutils.make_grid(input_, padding=2, nrow=nrow, pad_value=1).numpy().transpose((1,2,0))
        var_list.append(input_image)
    save_image = (np.vstack(var_list)*255).astype(np.uint8)
    save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2RGB)
    for index, each_title in enumerate(title_list):
        cv2.putText(save_image, each_title, (5, index*256+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0, 0, 255))
    cv2.imwrite(path, save_image)


CategoryOfStroke_to7 = [0,
                         1,
                         2,
                         3, 3,
                         4, 4, 5,
                         5,  5, 5, 5, 5, 5, 5, 5,
                         6, 6, 6, 2,
                         4, 4, 4, 2]


def seg_label_to7(id):
    return CategoryOfStroke_to7[id]


