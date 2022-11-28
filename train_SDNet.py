import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import time
import torch.optim as optim
import torch.utils.data as data
from load_data_for_SDNet import SDNetLoader
from model.model_of_SDNet import SDNet
from utils_loss_val import gradient_loss, ContentLoss, get_centroid_box_qualitative_result
from utils import save_picture, apply_stroke, seg_colors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt





class TrainSDNet():
    '''
        train SDNet with the Train-Dataset
        validate SDNet with the Test-Dataset
    '''

    def __init__(self, save_path=None, dataset=None):
        super().__init__()
        self.Out_path_train = os.path.join(save_path+'_'+dataset, 'train')
        self.Out_path_val = os.path.join(save_path + '_' + dataset, 'val')
        self.Model_path = os.path.join(save_path+'_'+dataset, 'model')
        self.Out_path_loss = os.path.join(save_path+'_'+dataset, 'loss')
        if not os.path.exists(self.Out_path_val):
            os.makedirs(self.Out_path_val)
        if not os.path.exists(self.Model_path):
            os.makedirs(self.Model_path)
        if not os.path.exists(self.Out_path_train):
            os.makedirs(self.Out_path_train)
        if not os.path.exists(self.Out_path_loss):
            os.makedirs(self.Out_path_loss)
        self.dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'dataset', dataset)

        # ContentLoss
        self.content_loss = ContentLoss().cuda()

        # Net
        self.sd_net = SDNet()
        self.sd_net.cuda()

    def save_model_parameter(self, epoch):
        # save models
        state_stn = {'net': self.sd_net.state_dict(), 'start_epoch': epoch}
        torch.save(state_stn, os.path.join(self.Model_path, 'sdnet_model.pth'))

    def train_model(self, epochs=40,  batch_size=16, init_learning_rate=0.001):
        self.batch_size = batch_size
        train_loader = data.DataLoader(SDNetLoader(is_training=True, dataset_path=self.dataset), batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(SDNetLoader(is_training=False, dataset_path=self.dataset), batch_size=batch_size)

        optim_op = optim.Adam(self.sd_net.parameters(), lr=init_learning_rate, betas=(0.5, 0.999))
        lr_scheduler_op = optim.lr_scheduler.ExponentialLR(optim_op, gamma=0.5)

        train_history_loss = []
        test_history_loss = []

        for i in range(epochs):
            print("Start training the %d epoch" % (i + 1))
            train_loss, loss_name = self.__train_epoch(i, train_loader, optim_op)
            test_loss, loss_name = self.__val_epoch(i, test_loader)
            # save loss
            train_history_loss.append(train_loss)
            test_history_loss.append(test_loss)
            for index, name in enumerate(loss_name):
                train_data = [x[index] for x in train_history_loss]
                test_data = [x[index] for x in test_history_loss]
                self.__plot_loss(name+'.png', [train_data, test_data], legend=['train', 'test'])
            # save model
            self.save_model_parameter(i)
            if (i+1)%10 == 0:
                lr_scheduler_op.step()

    def __plot_loss(self, name, loss, legend, save=True):
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
            save_path = os.path.join(self.Out_path_loss, name)
            plt.savefig(save_path)
        else:
            plt.show()

    def __calculate_linear_transformation_and_loss(self, reference_single_stroke, target_single_stroke, grid_for_linear, stroke_num,
                                                   reference_single_stroke_centroid):
        '''
        In Model Training and Testing,
        calculate local linear estimation transformation and transformed single target stroke.
        And, calculate content loss between transformed single target stroke and reference single stroke
        '''
        content_loss = torch.Tensor([0]).float().cuda()

        linear_tran_batch = []
        for i in range(self.batch_size):
            linear_tran_all = []
            loss__ = torch.Tensor([0]).float().cuda()
            num = 0
            for j in range(int(stroke_num[i])):
                target_stroke_single_ = target_single_stroke[i][j].unsqueeze(0).unsqueeze(0)
                reference_stroke_single_ = reference_single_stroke[i][j].unsqueeze(0).unsqueeze(0)
                grid_ = grid_for_linear[i].unsqueeze(0)
                affine_grid = self.sd_net.get_linear_estimation(reference_stroke_single_[0], grid_[0], reference_single_stroke_centroid[i][j])
                affine_tran = F.grid_sample(target_stroke_single_, affine_grid)
                c_loss_ = self.content_loss(affine_tran, reference_stroke_single_)
                linear_tran_all.append(affine_tran)
                num += 1
                loss__ += c_loss_
            if num > 0:
                content_loss = content_loss + loss__ / num
            affine_tran_whole_ = torch.clip(torch.sum(torch.cat(linear_tran_all, dim=1), dim=1, keepdim=True), 0, 1)
            linear_tran_batch.append(affine_tran_whole_)
        linear_tran_batch = torch.cat(linear_tran_batch, dim=0)
        return content_loss / self.batch_size, linear_tran_batch

    def __calculate_linear_transformation_inference(self, reference_single_stroke, target_single_stroke, grid_for_linear, stroke_num,
                                                   reference_single_stroke_centroid, seg_id):
        '''
        In Model Inference,
        calculate inverse local linear estimation transformation and transformed single reference stroke.
        '''
        transformed_reference_color = []
        transformed_single_reference_stroke = []
        for i in range(self.batch_size):
            linear_tran_whole = []
            for j in range(int(stroke_num[i])):
                target_stroke_single_ = target_single_stroke[i][j].unsqueeze(0).unsqueeze(0)
                reference_single_stroke_ = reference_single_stroke[i][j].unsqueeze(0).unsqueeze(0)
                grid_ = grid_for_linear[i].unsqueeze(0)
                linear_grid = self.sd_net.get_linear_estimation(reference_single_stroke_[0], grid_[0],
                                                                reference_single_stroke_centroid[i][j], inverse=True)
                linear_tran = F.grid_sample(reference_single_stroke_, linear_grid)
                linear_tran_whole.append(linear_tran)
            transformed_single_reference_stroke.append(torch.cat(linear_tran_whole, dim=1).squeeze(0))

            linear_tran_whole = self.__get_color_image(
                torch.cat(linear_tran_whole, dim=1).squeeze(0).detach().to('cpu').numpy()
                , seg_id[i, :int(stroke_num[i])].detach().to('cpu').numpy())

            transformed_reference_color.append(linear_tran_whole)
        transformed_reference_color = torch.cat(transformed_reference_color, dim=0)

        return transformed_reference_color, transformed_single_reference_stroke

    def __train_epoch(self, epoch, train_loader, optim_op):
        epoch += 1
        self.sd_net.train()

        loss_list = []
        start_time = time.time()

        for i, batch_sample in enumerate(train_loader):
            if batch_sample['target_data'].size(0) != self.batch_size:
                print('Batch size error!')
                continue
            # get data
            target_single_stroke = batch_sample['target_single_stroke'].cuda().float()
            reference_single_stroke = batch_sample['reference_single_stroke'].cuda().float()
            target_data = batch_sample['target_data'].cuda().float()
            reference_color = batch_sample['reference_color'].cuda().float()
            stroke_num = batch_sample['stroke_num'].cuda().float()
            reference_single_stroke_centroid = batch_sample['reference_single_stroke_centroid'].cuda().float()
            reference_image = torch.clip(torch.sum(reference_single_stroke, dim=1, keepdim=True), 0, 1)

            transformed_target_data, flow_global, grid_for_linear = self.sd_net.get_two_registration_field(reference_color, target_data)

            # calculate loss
            smooth_loss_global = gradient_loss(flow_global)
            content_loss_global = self.content_loss(transformed_target_data, reference_image)

            content_loss_single_linear, transformed_single_target_data_batch = self.__calculate_linear_transformation_and_loss(reference_single_stroke,
                                                                                                  target_single_stroke,
                                                                                                  grid_for_linear,
                                                                                                  stroke_num,
                                                                                                  reference_single_stroke_centroid)

            loss_sum = 0.5*content_loss_single_linear + content_loss_global + 5*smooth_loss_global
            loss_sum.backward()
            optim_op.step()
            optim_op.zero_grad()
            torch.cuda.empty_cache()
            loss_list.append([loss_sum.item(),  content_loss_global.item(), content_loss_single_linear.item(), smooth_loss_global.item()])

            if (i+1)%100==0 :
                save_list = []
                title_list = []
                title_list.append('reference_color')
                title_list.append('target_data')
                title_list.append('transformed_target_data')
                title_list.append('transformed_single_target_data_batch')

                save_list.append(reference_color.detach().to('cpu'))
                save_list.append(target_data.detach().to('cpu'))
                save_list.append(transformed_target_data.detach().to('cpu'))
                save_list.append(transformed_single_target_data_batch.detach().to('cpu'))

                save_picture(*save_list, title_list=title_list,
                             path=os.path.join(self.Out_path_train, str(epoch) + '_' + str(i) + '.bmp'),
                             nrow=int(save_list[0].size(0)))
                loss_value = np.mean(np.array(loss_list), axis=0)
                print(
                    "[TRAIN][{}/{}], loss_sum={:.7f},  content_loss_global={:.7f}, content_loss_single_linear={:.7f}, smooth_loss_global={:.7f}, time={:.3f}".format(
                        i, len(train_loader), loss_value[0], loss_value[1], loss_value[2],
                        loss_value[3], time.time() - start_time))
        loss_value = np.mean(np.array(loss_list), axis=0)
        loss_name = ['loss_sum', 'content_loss_global', 'content_loss_single_linear', 'smooth_loss_global']
        return loss_value, loss_name

    def __val_epoch(self, epoch, test_loader):
        epoch += 1
        self.sd_net.eval()


        loss_list = []

        start_time = time.time()
        for i, batch_sample in enumerate(test_loader):
            if batch_sample['target_data'].size(0) != self.batch_size:
                print('Batch size error!')
                continue
            # get data
            target_single_stroke = batch_sample['target_single_stroke'].cuda().float()
            reference_single_stroke = batch_sample['reference_single_stroke'].cuda().float()
            target_data = batch_sample['target_data'].cuda().float()
            reference_color = batch_sample['reference_color'].cuda().float()
            stroke_num = batch_sample['stroke_num'].cuda().float()
            reference_single_stroke_centroid = batch_sample['reference_single_stroke_centroid'].cuda().float()
            reference_image = torch.clip(torch.sum(reference_single_stroke, dim=1, keepdim=True), 0, 1)

            transformed_target_data, flow_global, grid_for_linear = self.sd_net.get_two_registration_field(reference_color, target_data)

            # calculate loss
            smooth_loss_global = gradient_loss(flow_global)
            content_loss_global = self.content_loss(transformed_target_data, reference_image)

            content_loss_single_linear, transformed_single_target_data_batch = self.__calculate_linear_transformation_and_loss(
                reference_single_stroke,
                target_single_stroke,
                grid_for_linear,
                stroke_num,
                reference_single_stroke_centroid)

            loss_sum = 0.5 * content_loss_single_linear + content_loss_global + 5 * smooth_loss_global
            loss_list.append([loss_sum.item(), content_loss_global.item(), content_loss_single_linear.item(),
                              smooth_loss_global.item()])

            if (i+1) % 2 == 0 and epoch%10==0:
                save_list = []
                title_list = []
                title_list.append('reference_color')
                title_list.append('target_data')
                title_list.append('transformed_target_data')
                title_list.append('transformed_single_target_data_batch')

                save_list.append(reference_color.detach().to('cpu'))
                save_list.append(target_data.detach().to('cpu'))
                save_list.append(transformed_target_data.detach().to('cpu'))
                save_list.append(transformed_single_target_data_batch.detach().to('cpu'))

                save_picture(*save_list, title_list=title_list,
                             path=os.path.join(self.Out_path_val, str(epoch) + '_' + str(i) + '.bmp'),
                             nrow=int(save_list[0].size(0)))
        loss_value = np.mean(np.array(loss_list), axis=0)
        loss_name = ['loss_sum', 'content_loss_global', 'content_loss_single_linear', 'smooth_loss_global']
        print(
            "[TEST][{}/{}], loss_sum={:.7f},  content_loss_global={:.7f}, content_loss_single_linear={:.7f}, smooth_loss_global={:.7f}, time={:.3f}".format(
                i, len(test_loader), loss_value[0], loss_value[1], loss_value[2],
                loss_value[3], time.time() - start_time))
        return loss_value, loss_name

    def __get_color_image(self, single_image, seg_id):
        color_kaiti = np.zeros(shape=(256, 256, 3))
        for i in range(single_image.shape[0]):
            color_kaiti = apply_stroke(color_kaiti, single_image[i], seg_colors[seg_id[i]])
        color_kaiti = np.transpose(color_kaiti, [2, 0, 1])
        return torch.from_numpy(color_kaiti).float().cuda().unsqueeze(0)

    def __save_data(self, save_path, save_num, transformed_reference_color, target_data,
                    transformed_reference_single_stroke, target_single_stroke, seg_id):
        '''
        save prior information and other data for training SegNet and ExtractNet
        '''
        style_image_save = torch.zeros(size=(7, 256, 256)).cuda().float()
        tran_kaiti_color_save = transformed_reference_color[0].detach().to('cpu').numpy()
        style_original_image_save = target_data[0]
        save_data = torch.cat([style_original_image_save, style_image_save], dim=0).detach().to(
            'cpu').numpy()
        np.save(os.path.join(save_path, str(save_num) + '_kaiti_color.npy'),
                tran_kaiti_color_save)
        np.save(os.path.join(save_path, str(save_num) + '_style.npy'), save_data > 0.5)

        single_tran_save = transformed_reference_single_stroke[0].detach().to('cpu').numpy()
        seg_id_save = seg_id.detach().to('cpu').numpy()
        np.save(os.path.join(save_path, str(save_num) + '_single.npy'),
                single_tran_save > 0.5)
        np.save(os.path.join(save_path, str(save_num) + '_seg.npy'), seg_id_save)
        style_single_image_save = target_single_stroke.detach().to('cpu').numpy()
        np.save(os.path.join(save_path, str(save_num) + '_style_single.npy'),
                style_single_image_save > 0.5)

    def calculate_prior_information_and_qualitative(self, save_path):
        '''
        In Model Inference,
        get prior information and other data for training SegNet and ExtractNet.
        And, calculate  qualitative result of mDis and mBIou
        :param batch_size:
        :param dataset:
        :return:
        '''
        save_train_dataset_path = os.path.join(save_path, 'train')
        save_test_dataset_path = os.path.join(save_path, 'test')
        save_qualitative_result_path = os.path.join(save_path, 'qualitative_result.txt')
        if not os.path.exists(save_train_dataset_path):
            os.makedirs(save_train_dataset_path)
        if not os.path.exists(save_test_dataset_path):
            os.makedirs(save_test_dataset_path)

        train_loader = data.DataLoader(SDNetLoader(is_training=True, dataset_path=self.dataset, is_inference=True), batch_size=1)
        test_loader = data.DataLoader(SDNetLoader(is_training=False, dataset_path=self.dataset, is_inference=True),batch_size=1)
        self.sd_net.eval()
        self.batch_size = 1

        # calculate with Train Dataset
        qualitative_result_list = []
        for j, batch_sample in enumerate(train_loader):
            if batch_sample['target_data'].size(0) != 1:
                print('Batch size error!')
                continue
            # get data
            target_single_stroke = batch_sample['target_single_stroke'].cuda().float()
            reference_single_stroke = batch_sample['reference_single_stroke'].cuda().float()
            target_data = batch_sample['target_data'].cuda().float()
            reference_color = batch_sample['reference_color'].cuda().float()
            stroke_num = batch_sample['stroke_num'].cuda().float()
            reference_single_stroke_centroid = batch_sample['reference_single_stroke_centroid'].cuda().float()
            seg_id = batch_sample['seg_id'].cuda().long()

            transformed_target_data, flow_global, grid_for_linear = self.sd_net.get_two_registration_field(
                reference_color, target_data)

            transformed_reference_color, transformed_single_reference_stroke = self.__calculate_linear_transformation_inference(
                reference_single_stroke,
                target_single_stroke,
                grid_for_linear, stroke_num,
                reference_single_stroke_centroid, seg_id)

            stroke_num = int(stroke_num[0])
            transformed_single_reference_stroke_ = transformed_single_reference_stroke[0].squeeze(1)
            target_single_stroke_ = target_single_stroke[0, :stroke_num]
            mDis, mBiou = get_centroid_box_qualitative_result(target_single_stroke_.detach().to('cpu').numpy(),
                                                              transformed_single_reference_stroke_.detach().to(
                                                                  'cpu').numpy())
            qualitative_result_list.append([mDis, mBiou])

            self.__save_data(save_train_dataset_path, j, transformed_reference_color,
                             target_data, transformed_single_reference_stroke, target_single_stroke_,
                             seg_id[0, :stroke_num])

        # save qualitative result
        loss_value = np.mean(np.array(qualitative_result_list), axis=0)
        loss_str = "[Train], mDis={:.7f},  mBIou={:.7f}".format(loss_value[0], loss_value[1])
        with open(save_qualitative_result_path, 'a+') as f:
            f.write(loss_str)

        # calculate with Test Dataset
        qualitative_result_list = []
        for j, batch_sample in enumerate(test_loader):
            if batch_sample['target_data'].size(0) != 1:
                print('Batch size error!')
                continue
            # get data
            target_single_stroke = batch_sample['target_single_stroke'].cuda().float()
            reference_single_stroke = batch_sample['reference_single_stroke'].cuda().float()
            target_data = batch_sample['target_data'].cuda().float()
            reference_color = batch_sample['reference_color'].cuda().float()
            stroke_num = batch_sample['stroke_num'].cuda().float()
            reference_single_stroke_centroid = batch_sample['reference_single_stroke_centroid'].cuda().float()
            seg_id = batch_sample['seg_id'].cuda().long()

            transformed_target_data, flow_global, grid_for_linear = self.sd_net.get_two_registration_field(
                reference_color, target_data)

            transformed_reference_color, transformed_single_reference_stroke = self.__calculate_linear_transformation_inference(reference_single_stroke,
                                                                                                  target_single_stroke,
                                                                            grid_for_linear, stroke_num,
                                                                            reference_single_stroke_centroid,seg_id)


            stroke_num = int(stroke_num[0])
            transformed_single_reference_stroke_ = transformed_single_reference_stroke[0].squeeze(1)
            target_single_stroke_ = target_single_stroke[0, :stroke_num]
            mDis, mBiou = get_centroid_box_qualitative_result(target_single_stroke_.detach().to('cpu').numpy(),
                                                         transformed_single_reference_stroke_.detach().to('cpu').numpy())
            qualitative_result_list.append([mDis, mBiou])

            self.__save_data(save_test_dataset_path, j, transformed_reference_color,
                             target_data, transformed_single_reference_stroke, target_single_stroke_, seg_id[0, :stroke_num])

        # save qualitative result
        loss_value = np.mean(np.array(qualitative_result_list), axis=0)
        loss_str = "[TEST], mDis={:.7f},  mBIou={:.7f}".format(loss_value[0], loss_value[1])
        with open(save_qualitative_result_path, 'a+') as f:
            f.write(loss_str)


if __name__ == '__main__':
    model = TrainSDNet(save_path='out/SDNet', dataset='CCSEDB')
    model.train_model(epochs=40, init_learning_rate=0.0001, batch_size=8, dataset=r'dataset\CCSEDB')
    model.calculate_prior_information_and_qualitative('dataset_forSegNet_ExtractNet_CCSEDB')


