import os.path
import torch.utils.data as data
import pickle
import time
import torch
import scipy.ndimage as pyimg
import pandas as pd
import os
import os.path
import glob
import matplotlib.pyplot as plt
import numpy as np
from utils import seg_label_to7

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''
load data for SegNet and ExtractNet
'''


class SegNetExtractNetLoader(data.Dataset):
    def __init__(self, is_training, dataset_path, is_single=False):
        self.is_training = is_training
        self.is_single = is_single
        if is_training:
            self.path = [[os.path.join(dataset_path, 'train', each),
                          os.path.join(dataset_path, 'train', each[:-16] + '_style.npy'),
                          os.path.join(dataset_path, 'train', each[:-16] + '_seg.npy'),
                          os.path.join(dataset_path, 'train', each[:-16] + '_single.npy'),
                          os.path.join(dataset_path, 'train', each[:-16] + '_style_single.npy')]
                           for each in os.listdir(os.path.join(dataset_path, 'train')) if 'color' in each]
        else:
            self.path = [[os.path.join(dataset_path, 'test', each),
                          os.path.join(dataset_path, 'test', each[:-16] + '_style.npy'),
                          os.path.join(dataset_path, 'test', each[:-16] + '_seg.npy'),
                          os.path.join(dataset_path, 'test', each[:-16] + '_single.npy'),
                          os.path.join(dataset_path, 'test', each[:-16] + '_style_single.npy'), int(each[:-16])]
                           for each in os.listdir(os.path.join(dataset_path, 'test')) if
                         'color' in each]

        self.path = sorted(self.path, key=lambda x: x[-1])
        print("number of dataset：%d"%len(self.path))

    def get_seg_image(self, reference_single, seg_label):
        reference_image = np.zeros(shape=(7, 256, 256), dtype=np.float)
        for i in range(seg_label.shape[0]):
            id_7 = seg_label_to7(seg_label[i])
            reference_image[id_7] += reference_single[i]
        return np.clip(reference_image, 0, 1)

    def get_data(self, item):
        """
        """
        reference_color = np.load(self.path[item][0])  # (3, 256, 256)
        label_seg = np.load(self.path[item][1])[1:]  # (7, 256, 256)
        target_image = np.load(self.path[item][1])[:1]  # (1, 256, 256)
        seg_id = np.load(self.path[item][2])     # (N)
        reference_transformed_single = np.load(self.path[item][3])  # (N, 256 256)
        target_single_stroke = np.load(self.path[item][4])  # (N, 256 256)
        target_data = np.repeat(target_image, 3, axis=0).astype(np.float)
        reference_segment_transformation_data = self.get_seg_image(reference_transformed_single, seg_id)
        label_seg = self.get_seg_image(target_single_stroke, seg_id)


        if self.is_single:  # For ExtractNet
            return {
                'target_data': target_data,
                'reference_color':reference_color,
                'label_seg': label_seg,
                'reference_segment_transformation_data':reference_segment_transformation_data,
                'seg_id': seg_id,
                'reference_transformed_single': reference_transformed_single,
                'target_single_stroke': target_single_stroke

            }
        else:  # For SegNet
            return {

                'target_data': target_data,
                'reference_color': reference_color,
                'label_seg': label_seg
            }

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        data = self.get_data(item)
        return data



