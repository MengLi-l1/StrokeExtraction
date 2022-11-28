import os.path
import torch.utils.data as data
import pickle
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

'''
Load input data for SDNet
'''


class SDNetLoader(data.Dataset):
    def __init__(self, is_training, dataset_path, is_inference=False):
        self.is_inference = is_inference
        if is_training:
            self.path = [[os.path.join(dataset_path, 'train', each),
                          os.path.join(dataset_path, 'train', each[:-16] + '_original_style.npy'),
                          os.path.join(dataset_path, 'train', each[:-16] + '_kaiti_center.npy'),
                          os.path.join(dataset_path, 'train', each[:-16] + '_seg.npy'),
                          os.path.join(dataset_path, 'train', each[:-16] + '_kaiti_single.npy'),
                          os.path.join(dataset_path, 'train', each[:-16] + '_style_single.npy')]
                           for each in os.listdir(os.path.join(dataset_path, 'train')) if 'color' in each]
        else:
            self.path = [[os.path.join(dataset_path, 'test', each),
                          os.path.join(dataset_path, 'test', each[:-16] + '_original_style.npy'),
                          os.path.join(dataset_path, 'test', each[:-16] + '_kaiti_center.npy'),
                          os.path.join(dataset_path, 'test', each[:-16] + '_seg.npy'),
                          os.path.join(dataset_path, 'test', each[:-16] + '_kaiti_single.npy'),
                          os.path.join(dataset_path, 'test', each[:-16] + '_style_single.npy')]
                           for each in os.listdir(os.path.join(dataset_path, 'test')) if
                         'color' in each]
        print("number of dataset：%d" % len(self.path))

    def get_data(self, item):
        """
        """
        kaiti_color = np.load(self.path[item][0])  # (3, 256, 256)
        original_style = np.load(self.path[item][1])  # (1, 256, 256)
        seg_id = np.load(self.path[item][3])  # (N)
        kaiti_single = np.load(self.path[item][4])  # (N, 256 256)
        style_single = np.load(self.path[item][5])  # (N, 256 256)

        stroke_num = kaiti_single.shape[0]
        expand_zeros = []
        kaiti_single_image_center = []

        for i in range(30):
            if i >= kaiti_single.shape[0]:
                expand_zeros.append(np.zeros(shape=(256, 256), dtype=np.float))
                kaiti_single_image_center.append(np.array([127.5, 127.5]))

            else:
                point = np.where(kaiti_single[i] > 0.5)
                # Calculate
                center = np.array([np.mean(point[1]), np.mean(point[0])])
                kaiti_single_image_center.append(center)
        expand_zeros = np.array(expand_zeros)
        kaiti_single = np.concatenate([kaiti_single, expand_zeros], axis=0)
        style_single = np.concatenate([style_single, expand_zeros], axis=0)
        kaiti_center = np.array(kaiti_single_image_center)


        if not self.is_inference:
            return {
                'target_single_stroke': style_single,
                'reference_single_stroke': kaiti_single,
                'target_data': original_style,
                'reference_color': kaiti_color,
                'stroke_num': stroke_num,
                'reference_single_stroke_centroid': kaiti_center,
            }
        else:
            return {
                'target_single_stroke': style_single,
                'reference_single_stroke': kaiti_single,
                'target_data': original_style,
                'reference_color': kaiti_color,
                'stroke_num': stroke_num,
                'reference_single_stroke_centroid': kaiti_center,
                'seg_id': seg_id
            }




    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        data = self.get_data(item)
        return data


def save_dict(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    c = SDNetLoader(is_training=False, dataset_path='dataset/CCSEDB')
    data_loader = data.DataLoader(c, batch_size=8, shuffle=False)

    for i in range(10):
        print(i)
        for i_batch, sample_batched in enumerate(data_loader):

            style_single_image = sample_batched['style_single_image'][0].numpy()
            kaiti_single_image = sample_batched['kaiti_single_image'][0].numpy()
            original = sample_batched['original_style'].numpy()
            kaiti_color = sample_batched['kaiti_color'].numpy()
            stroke_num = sample_batched['stroke_num'][0].numpy()
            kaiti_center = sample_batched['kaiti_center'].numpy()
            assert (not np.isnan(kaiti_center).any())
            stroke_num = int(stroke_num)
            plt.figure(0)
            for j in range(8):
                plt.subplot2grid((2, 8), (0, j))
                plt.imshow(kaiti_color[j].transpose((1, 2, 0)))
                plt.subplot2grid((2, 8), (1, j))
                plt.imshow(original[j].squeeze())
            plt.show()
            # plt.figure(1)
            #
            # for j in range(stroke_num):
            #     plt.subplot2grid((2, stroke_num), (0, j))
            #     plt.imshow(kaiti_single_image[j])
            #     plt.subplot2grid((2, stroke_num), (1, j))
            #     plt.imshow(style_single_image[j])
            #
            # plt.show()

            # seg_id = sample_batched['seg_id'][0].numpy()
            #
            # data = { 'style_single_image': style_single_image,
            # 'kaiti_single_image': kaiti_single_image,
            # 'style_image': style_image,
            # 'kaiti_image': kaiti_image,
            # 'seg_id':seg_id}
            # save_dict(data, os.path.join(path, str(train_num)+'.pkl'))
            # train_num+=1
            # print(train_num)


