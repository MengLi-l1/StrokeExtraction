import torch
import torch.nn.functional as F
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.optim as optim
import torch.utils.data as data
from model.model_of_ExtractNet import ExtractNet
from model.model_of_SegNet import SegNet
from load_data_for_SegNetExtractNet import SegNetExtractNetLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
from utils import random_colors, apply_stroke_t, save_picture, seg_colors, apply_stroke, seg_label_to7
from utils_loss_val import get_iou_without_matching, get_iou_with_matching


class DataPool(object):
    '''
    In actual situations, due to the different number of strokes in each character, in order to achieve batch training,
    we build a data pool
    '''
    def __init__(self):
        self.data_num = 0
        self.pool_data = {}

    def add(self, target_data, reference_stroke_transformation_data, segment_data,
            reference_segment_transformation_data, segNet_feature, label, cut_box_list):
        '''
        put data into pool
        '''

        if 'target_data' not in self.pool_data.keys():
            self.pool_data['target_data'] = target_data
        else:
            self.pool_data['target_data'] = torch.cat([self.pool_data['target_data'], target_data],
                                                           dim=0)

        if 'reference_stroke_transformation_data' not in self.pool_data.keys():
            self.pool_data['reference_stroke_transformation_data'] = reference_stroke_transformation_data
        else:
            self.pool_data['reference_stroke_transformation_data'] = torch.cat([self.pool_data['reference_stroke_transformation_data'], reference_stroke_transformation_data], dim=0)

        if 'segment_data' not in self.pool_data.keys():
            self.pool_data['segment_data'] = segment_data
        else:
            self.pool_data['segment_data'] = torch.cat(
                [self.pool_data['segment_data'], segment_data], dim=0)

        if 'reference_segment_transformation_data' not in self.pool_data.keys():
            self.pool_data['reference_segment_transformation_data'] = reference_segment_transformation_data
        else:
            self.pool_data['reference_segment_transformation_data'] = torch.cat([self.pool_data['reference_segment_transformation_data'], reference_segment_transformation_data],
                                                              dim=0)

        if 'segNet_feature' not in self.pool_data.keys():
            self.pool_data['segNet_feature'] = segNet_feature
        else:
            self.pool_data['segNet_feature'] = torch.cat([self.pool_data['segNet_feature'], segNet_feature], dim=0)

        if 'label' not in self.pool_data.keys():
            self.pool_data['label'] = label
        else:
            self.pool_data['label'] = torch.cat([self.pool_data['label'], label], dim=0)

        if 'cut_box_list' not in self.pool_data.keys():
            self.pool_data['cut_box_list'] = cut_box_list
        else:
            self.pool_data['cut_box_list'].extend(cut_box_list)

        self.data_num += int(label.size(0))

    def next(self, num):
        # get next training data from pool

        target_data_batch = self.pool_data['target_data'][:num]
        self.pool_data['target_data'] = self.pool_data['target_data'][num:]

        reference_stroke_transformation_data_batch = self.pool_data['reference_stroke_transformation_data'][:num]
        self.pool_data['reference_stroke_transformation_data'] = self.pool_data['reference_stroke_transformation_data'][num:]

        segment_data_batch = self.pool_data['segment_data'][:num]
        self.pool_data['segment_data'] = self.pool_data['segment_data'][num:]

        reference_segment_transformation_data_batch = self.pool_data['reference_segment_transformation_data'][:num]
        self.pool_data['reference_segment_transformation_data'] = self.pool_data['reference_segment_transformation_data'][num:]

        segNet_feature_batch = self.pool_data['segNet_feature'][:num]
        self.pool_data['segNet_feature'] = self.pool_data['segNet_feature'][num:]

        label_batch = self.pool_data['label'][:num]
        self.pool_data['label'] = self.pool_data['label'][num:]

        cut_list_batch = self.pool_data['cut_box_list'][:num]
        self.pool_data['cut_box_list'] = self.pool_data['cut_box_list'][num:]

        self.data_num -= num

        return [target_data_batch, reference_stroke_transformation_data_batch, segment_data_batch,
        reference_segment_transformation_data_batch, segNet_feature_batch, label_batch, cut_list_batch]





class TrainExtractNet():
    '''
        train SDNet with the Train-Dataset
        validate SDNet with the Test-Dataset
    '''

    def __init__(self,  save_path=None, segNet_save_path=None):
        super().__init__()
        self.segNet_save_path = segNet_save_path
        self.Out_path_train = os.path.join(save_path, 'train')
        self.Model_path = os.path.join(save_path, 'model')
        self.Out_path_loss = os.path.join(save_path, 'loss')
        self.Out_path_val = os.path.join(save_path, 'val')
        self.Out_path_train_stage2 = os.path.join(save_path, 'train_stage2')
        self.Out_path_test_stage2 = os.path.join(save_path, 'test_stage2')
        self.Out_path_loss_stage2 = os.path.join(save_path, 'loss_stage2')
        if not os.path.exists(self.Model_path):
            os.makedirs(self.Model_path)
        if not os.path.exists(self.Out_path_train):
            os.makedirs(self.Out_path_train)
        if not os.path.exists(self.Out_path_loss):
            os.makedirs(self.Out_path_loss)
        if not os.path.exists(self.Out_path_val):
            os.makedirs(self.Out_path_val)

        # SegNet
        self.seg_net = SegNet(out_feature=True)


        # Extract
        self.extract_net = ExtractNet()
        self.extract_net.cuda()

        # Data Pool
        self.data_pool = DataPool()

    def save_model_parameter(self, epoch):
        # save models
        state_stn = {'net': self.extract_net.state_dict(), 'start_epoch': epoch}
        torch.save(state_stn, os.path.join(self.Model_path, 'model_extract.pth'))

    def train_model(self, epochs=40, batch_size=16, init_learning_rate=0.001, dataset=None):
        self.batch_size = batch_size

        # load parameters of SegNet
        seg_model_path = os.path.join(self.segNet_save_path, 'model', 'model.pth')
        state = torch.load(seg_model_path)
        self.seg_net.load_state_dict(state['net'])
        self.seg_net.to('cuda').eval().requires_grad_(False)

        # dataset
        train_loader = data.DataLoader(SegNetExtractNetLoader(is_training=True, dataset_path=dataset, is_single=True), batch_size=1, shuffle=True)
        test_loader = data.DataLoader(SegNetExtractNetLoader(is_training=False, dataset_path=dataset, is_single=True), batch_size=1)
        optim_op = optim.Adam(self.extract_net.parameters(), lr=init_learning_rate, betas=(0.5, 0.999))
        lr_scheduler_op = optim.lr_scheduler.ExponentialLR(optim_op, gamma=0.5)

        train_history_loss = []
        test_history_loss = []

        for i in range(epochs):
            print("Start training the %d epoch" % (i + 1))
            train_loss, loss_name = self.__train_epoch(i, train_loader, optim_op)
            test_loss, loss_name = self.__val_epoch(i, test_loader)
            train_history_loss.append(train_loss)
            test_history_loss.append(test_loss)
            for index, name in enumerate(loss_name):
                train_data = [x[index] for x in train_history_loss]
                test_data = [x[index] for x in test_history_loss]
                self.__plot_loss(name+'stage2.png', [train_data, test_data],
                               legend=['train', 'test'], folder_name=self.Out_path_loss_stage2)
            # save models
            self.save_model_parameter(i)
            if (i+1)%5 == 0:
                lr_scheduler_op.step()

    def __plot_loss(self, name, loss, legend, folder_name, save=True):
        '''
        @param name: name
        @param loss: array,shape=(N, 2)
        @return:
        '''
        loss_ = np.array(loss)
        plt.figure("loss")
        plt.gcf().clear()

        for i in range(len(legend)):
            plt.plot(loss_[i,:], label=legend[i])

        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        if save:
            save_path = os.path.join(folder_name, name)
            plt.savefig(save_path)
        else:
            plt.show()

    def __get_cut_region(self, kaiti_imagae):
        if np.sum(kaiti_imagae) > 5:
            points = np.where(kaiti_imagae > 0.5)
            x_l = np.min(points[1])
            x_r = np.max(points[1])
            y_t = np.min(points[0])
            y_b = np.max(points[0])
        else:
            y_t = 0
            y_b = 255
            x_l = 0
            x_r = 255
        w = x_r - x_l + 1
        h = y_b - y_t + 1
        center_x = int((x_l+x_r)/2)
        center_y = int((y_t+y_b)/2)
        size = max(w, h)
        if size>32:
            cut_size = 256
            # y_t, y_b, x_l, x_r
            return [0, 256, 0, 256]
        elif size>16:
            cut_size = 128
        else:
            cut_size = 64

        if center_x - cut_size/2 <= 0:
            x_l = 0
            x_r = cut_size
        elif center_x+cut_size/2>=256:
            x_l = 256-cut_size
            x_r = 256
        else:
            x_l = int(center_x - cut_size/2)
            x_r = x_l + cut_size

        if center_y - cut_size/2 <= 0:
            y_t = 0
            y_b = cut_size
        elif center_y+cut_size/2>=256:
            y_t = 256-cut_size
            y_b = 256
        else:
            y_t = int(center_y - cut_size/2)
            y_b = y_t + cut_size

        return [y_t, y_b, x_l, x_r]

    def __create_color_image(self, image, id):
        image = image.repeat(1, 3, 1, 1)
        color = torch.from_numpy(np.array(seg_colors[id]).reshape(1, 3, 1, 1)).cuda()
        return image * color

    def __get_training_data_of_ExtarctNet(self, reference_transformed_single, target_single_stroke, seg_index,
                                   seg_out, reference_segment_transformation_data,
                                          target_data, seg_out_feature):
        '''
        get training_data of ExtractNet

        When the size of reference stroke is too small, the stroke area is clipped and enlarged
        to increase the discrimination ability.
        '''

        kaiti_tran_single_image = torch.reshape(reference_transformed_single, shape=(-1, 1, 256, 256))
        style_single_image = torch.reshape(target_single_stroke, shape=(-1, 1, 256, 256))
        kaiti_trans_stage2_in = []
        kaiti_trans_seg_stage2_in = []
        style_stage2_in = []
        seg_out_stage_in = []
        style_original_stage2_in = []
        cut_box_list = []
        for index in range(int(seg_index.size(0))):
            id = int(seg_index[index])
            id_7 = seg_label_to7(id)
            kaiti_single_image_trans = kaiti_tran_single_image[index:index + 1]

            # get cut region：
            kaiti_image__ = np.squeeze(kaiti_single_image_trans.to('cpu').numpy())
            cut_box = self.__get_cut_region(kaiti_image__)
            torch_resize = Resize([256, 256])
            y_t, y_b, x_l, x_r = cut_box
            cut_box_list.append(cut_box)

            # Reference Stroke Transformation Data
            kaiti_tran_in = kaiti_single_image_trans.detach()
            kaiti_tran_in = torch_resize(kaiti_tran_in[:, :, y_t:y_b, x_l:x_r])
            kaiti_tran_in = self.__create_color_image(kaiti_tran_in, id).float()
            kaiti_trans_stage2_in.append(kaiti_tran_in)

            # label
            style_in = torch.unsqueeze(style_single_image[index], dim=0).float()
            style_in = torch_resize(style_in[:, :, y_t:y_b, x_l:x_r])
            style_stage2_in.append(style_in)

            # Segment Data
            seg_out_in = F.sigmoid(seg_out[:, id_7:id_7 + 1]).detach().float()
            seg_out_in = torch_resize(seg_out_in[:, :, y_t:y_b, x_l:x_r])
            seg_out_stage_in.append(seg_out_in)

            # Reference Segment Transformation Data
            kaiti_seg_in = reference_segment_transformation_data[:, id_7:id_7 + 1].detach().float()
            kaiti_seg_in = torch_resize(kaiti_seg_in[:, :, y_t:y_b, x_l:x_r])
            kaiti_trans_seg_stage2_in.append(kaiti_seg_in)

            # Target Data
            style_image_original_in = torch_resize(target_data[:, :, y_t:y_b, x_l:x_r])
            style_original_stage2_in.append(style_image_original_in)
        reference_stroke_transformation_data = torch.cat(kaiti_trans_stage2_in, dim=0)
        label = torch.cat(style_stage2_in, dim=0)
        segment_data = torch.cat(seg_out_stage_in, dim=0)
        segNet_feature = seg_out_feature['out_64_32'].repeat(label.size(0), 1, 1, 1)
        target_data = torch.cat(style_original_stage2_in, dim=0)
        reference_segment_transformation_data = torch.cat(kaiti_trans_seg_stage2_in, dim=0)

        return [target_data, reference_stroke_transformation_data, segment_data,
                reference_segment_transformation_data, segNet_feature, label, cut_box_list]

    def __to_original_stroke(self, out, label, cut_box_list):
        '''
        Restore strokes to their original size
        :param cut_box_list: Adaptive size parameters
        :return:
        '''
        out = out.squeeze(1).to('cpu').numpy()
        label = label.squeeze(1).to('cpu').numpy()
        out_re = []
        label_re = []

        for i in range(len(cut_box_list)):
            y_t, y_b, x_l, x_r = cut_box_list[i]
            img = np.zeros_like(out[i])
            img[y_t:y_b, x_l:x_r] = cv2.resize(out[i].astype(np.uint8), dsize=(x_r - x_l, y_b - y_t))
            out_re.append(img)

            img_r = np.zeros_like(label[i])
            img_r[y_t:y_b, x_l:x_r] = cv2.resize(label[i].astype(np.uint8), dsize=(x_r - x_l, y_b - y_t))
            label_re.append(img_r)
        return out_re, label_re

    def __train_epoch(self, epoch, train_loader, optim_opWhole):
        epoch += 1
        self.extract_net.train()
        loss_list = []
        start_time = time.time()

        pool_i = 0

        for i, batch_sample in enumerate(train_loader):
            # get data
            reference_color = batch_sample['reference_color'].float().cuda()
            reference_segment_transformation_data = batch_sample['reference_segment_transformation_data'].float().cuda()
            target_data = batch_sample['target_data'].float().cuda()
            target_single_stroke = batch_sample['target_single_stroke'].float().cuda()
            reference_transformed_single = batch_sample['reference_transformed_single'].float().cuda()
            seg_index = batch_sample['seg_id'][0].long().cuda()

            # get segment result fo SegNet
            seg_out, seg_out_feature = self.seg_net(target_data, reference_color)

            # get inputs of ExtractNet
            target_data, reference_stroke_transformation_data, segment_data, \
            reference_segment_transformation_data, segNet_feature, label, cut_box_list = self.__get_training_data_of_ExtarctNet(reference_transformed_single, target_single_stroke, seg_index,
                                   seg_out, reference_segment_transformation_data,
                                          target_data, seg_out_feature)
            # put data into pool
            self.data_pool.add(target_data, reference_stroke_transformation_data, segment_data,
                                reference_segment_transformation_data, segNet_feature, label, cut_box_list)

            while self.data_pool.data_num >= self.batch_size:
                target_data_batch, reference_stroke_transformation_data_batch, segment_data_batch,\
                reference_segment_transformation_data_batch, segNet_feature_batch, label_batch, cut_box_list_batch = self.data_pool.next(self.batch_size)

                extract_out = self.extract_net(reference_stroke_transformation_data_batch,
                                                       reference_segment_transformation_data_batch, segment_data_batch,
                                                       target_data_batch, segNet_feature_batch)

                # calculate loss
                loss = F.binary_cross_entropy(F.sigmoid(extract_out), label_batch)
                extract_result = F.sigmoid(extract_out).detach() > 0.5

                #  Restore strokes to their original size
                # Calculate mIOUm and mIOUum
                #  Restore strokes to their original size
                extract_result, label = self.__to_original_stroke(extract_result, label_batch, cut_box_list_batch)
                mIOUm = get_iou_with_matching(extract_result, label)


                loss.backward()
                optim_opWhole.step()
                optim_opWhole.zero_grad()

                torch.cuda.empty_cache()
                loss_list.append([loss.item(),  mIOUm.item(), 0])




        loss_value = np.mean(np.array(loss_list), axis=0)
        loss_name = ['loss',  'mIOUm', 'mIOUum']

        print(
             "[TRAIN][{}/{}], loss={:.7f},  mIOUm={:.7f}, mIOUum={:.7f}, time={:.7f}".format(
                        i, len(train_loader), loss_value[0], loss_value[1], loss_value[2],time.time() - start_time))

        return loss_value, loss_name

    def __val_epoch(self, epoch, test_loader):
        epoch += 1
        self.extract_net.eval()
        loss_list = []
        start_time = time.time()

        for i, batch_sample in enumerate(test_loader):
            # get data
            reference_color = batch_sample['reference_color'].float().cuda()
            reference_segment_transformation_data = batch_sample['reference_segment_transformation_data'].float().cuda()
            target_data_o = batch_sample['target_data'].float().cuda()
            target_single_stroke = batch_sample['target_single_stroke'].float().cuda()
            reference_transformed_single = batch_sample['reference_transformed_single'].float().cuda()
            seg_index = batch_sample['seg_id'][0].long().cuda()

            # get segment result fo SegNet
            seg_out, seg_out_feature = self.seg_net(target_data_o, reference_color)

            # get inputs of ExtractNet
            target_data, reference_stroke_transformation_data, segment_data, \
            reference_segment_transformation_data, segNet_feature, label, cut_box_list = self.__get_training_data_of_ExtarctNet(
                reference_transformed_single, target_single_stroke, seg_index,
                seg_out, reference_segment_transformation_data,
                target_data_o, seg_out_feature)

            extract_out = self.extract_net(reference_stroke_transformation_data,
                                           reference_segment_transformation_data, segment_data,
                                           target_data, segNet_feature)

            # calculate loss
            loss = F.binary_cross_entropy(F.sigmoid(extract_out), label)
            extract_result = F.sigmoid(extract_out).detach() > 0.5

            #  Restore strokes to their original size
            extract_result, label = self.__to_original_stroke(extract_result, label, cut_box_list)

            # Calculate mIOUm and mIOUum
            mIOUm = get_iou_with_matching(extract_result, label)
            mIOUum = get_iou_without_matching(extract_result, label)

            loss.backward()
            torch.cuda.empty_cache()
            loss_list.append([loss.item(), mIOUm.item(), mIOUum.item()])
            if (i+1)%5==0 and (epoch+1)%1==0:
                # save data
                extract_result_show = np.zeros(shape=(256, 256, 3), dtype=np.float) + target_data_o.squeeze().detach().to(
                                                    'cpu').numpy().transpose(1, 2, 0)

                label_result_show = np.zeros(shape=(256, 256, 3),dtype=np.float) + target_data_o.squeeze().detach().to(
                                                        'cpu').numpy().transpose(1, 2, 0)

                r_colors = random_colors(len(extract_result))
                def shuffle_color(c, step=3):
                    shuf_c = []
                    for i in range(step):
                        num = 0
                        while i + num * step < len(c):
                            shuf_c.append(c[i + num * step])
                            num += 1
                    return shuf_c
                r_colors = shuffle_color(r_colors)
                for k in range(len(extract_result)):
                    extract_result_show = apply_stroke_t(extract_result_show, extract_result[k] > 0.5, r_colors[k])
                    label_result_show = apply_stroke_t(label_result_show, label[k] > 0.5, r_colors[k])
                save_list = []
                title_list = []
                title_list.append('reference_color')
                title_list.append('target_data')
                title_list.append('stroke_label')
                title_list.append('stroke_extraction')

                save_list.append(reference_color.detach().to('cpu').repeat(2, 1, 1, 1))
                save_list.append(target_data_o.detach().to('cpu').repeat(2, 1, 1, 1))
                save_list.append(torch.from_numpy(label_result_show.transpose(2, 0, 1)).unsqueeze(0).repeat(2, 1, 1, 1))
                save_list.append(torch.from_numpy(extract_result_show.transpose(2, 0, 1)).unsqueeze(0).repeat(2, 1, 1, 1))

                save_picture(*save_list, title_list=title_list,
                             path=os.path.join(self.Out_path_val, str(i) + '.bmp'),
                             nrow=int(save_list[0].size(0)))

        loss_value = np.mean(np.array(loss_list), axis=0)
        loss_name = ['loss', 'mIOUm', 'mIOUum']

        print(
            "[TEST][{}/{}], loss={:.7f},  mIOUm={:.7f}, mIOUum={:.7f}, time={:.7f}".format(
                i, len(test_loader), loss_value[0], loss_value[1], loss_value[2], time.time() - start_time))
        return loss_value, loss_name





if __name__ == '__main__':
    model = TrainExtractNet(save_path='out/ExtractNet_CCSEDB', segNet_save_path='out/SegNet_CCSEDB')
    model.train_model(epochs=20, init_learning_rate=0.0001, batch_size=8, dataset=r'dataset_forSegNet_ExtractNet_CCSEDB')
