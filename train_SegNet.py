import torch
import torch.nn.functional as F
import os
import time
import torch.optim as optim
import torch.utils.data as data
from model.model_of_SegNet import SegNet
from load_data_for_SegNetExtractNet import SegNetExtractNetLoader
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt
from utils import save_picture, random_colors, apply_stroke
from utils_loss_val import get_mean_IOU


seg_colors = random_colors(7)


class TrainSegNet():
    '''
        train SegNet with the Train-Dataset
        validate SegNet with the Test-Dataset
    '''

    def __init__(self,  save_path=None):
        super().__init__()
        self.Out_path_train = os.path.join(save_path, 'train')
        self.Model_path = os.path.join(save_path, 'model')
        self.Out_path_val = os.path.join(save_path, 'val')
        self.Out_path_loss = os.path.join(save_path, 'loss')

        if not os.path.exists(self.Model_path):
            os.makedirs(self.Model_path)
        if not os.path.exists(self.Out_path_train):
            os.makedirs(self.Out_path_train)
        if not os.path.exists(self.Out_path_loss):
            os.makedirs(self.Out_path_loss)
        if not os.path.exists(self.Out_path_val):
            os.makedirs(self.Out_path_val)

        # SegNet
        self.seg_net = SegNet(out_feature=False)
        self.seg_net.to('cuda')

    def save_model_parameter(self, epoch):
        # save models
        state_stn = {'net': self.seg_net.state_dict(), 'start_epoch': epoch}
        torch.save(state_stn, os.path.join(self.Model_path, 'model.pth'))

    def train_model(self, epochs=40,  batch_size=16, init_learning_rate=0.001, dataset_path = None):
        self.batch_size = batch_size
        train_loader = data.DataLoader(SegNetExtractNetLoader(is_training=True, dataset_path=dataset_path), batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(SegNetExtractNetLoader(is_training=False, dataset_path=dataset_path), batch_size=batch_size)  # 24涓敤浜庢祴璇?

        optim_op = optim.Adam(self.seg_net.parameters(), lr=init_learning_rate, betas=(0.5, 0.999))
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
                self.__plot_loss(name+'.png', [train_data, test_data],
                               legend=['train', 'test'])
            # save models
            self.save_model_parameter(i)
            if (i+1)%2 == 0:
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

    def __train_epoch(self, epoch, train_loader, optim_opWhole):
        epoch += 1
        self.seg_net.train()
        loss_list = []
        start_time = time.time()

        for i, batch_sample in enumerate(train_loader):
            if batch_sample['target_data'].size(0) != self.batch_size:
                print('Batch size error!')
                continue
            # get data
            reference_color = batch_sample['reference_color'].float().cuda()
            label_seg = batch_sample['label_seg'].float().cuda()
            target_data = batch_sample['target_data'].float().cuda()

            seg_out = self.seg_net(target_data, reference_color)

            seg_loss = F.binary_cross_entropy(F.sigmoid(seg_out), label_seg)
            seg_result_ = (F.sigmoid(seg_out).detach() > 0.5)
            mean_iou = get_mean_IOU(seg_result_, label_seg)
            optim_opWhole.zero_grad()
            seg_loss.backward()
            optim_opWhole.step()
            torch.cuda.empty_cache()
            loss_list.append([seg_loss.item(), mean_iou.item()])

            if i%50==0:
                save_list = []
                title_list = []
                title_list.append('reference_color')
                title_list.append('target_data')
                title_list.append('label_seg')
                title_list.append('seg_result')

                save_list.append(reference_color.detach().to('cpu'))
                save_list.append(target_data.detach().to('cpu'))

                save_list.append(self.__to_color(label_seg))
                save_list.append(self.__to_color(seg_result_))

                save_picture(*save_list, title_list=title_list,
                             path=os.path.join(self.Out_path_train, str(epoch) + '_' + str(i) + '.bmp'),
                             nrow=int(save_list[0].size(0)))


        loss_value = np.mean(np.array(loss_list), axis=0)
        loss_name = ['seg_loss', 'mean_iou']
        print(
            "[TRAIN][{}/{}],   seg_loss={:.7f}, mean_iou={:.7f}, time={:.7f}".format(i, len(train_loader), loss_value[0], loss_value[1], time.time() - start_time))

        return loss_value, loss_name

    def __val_epoch(self, epoch, test_loader):
        epoch += 1
        self.seg_net.eval()
        loss_list = []

        start_time = time.time()
        for i, batch_sample in enumerate(test_loader):
            if batch_sample['target_data'].size(0) != self.batch_size:
                print('Batch size error!')
                continue
            # get data
            reference_color = batch_sample['reference_color'].float().cuda()
            label_seg = batch_sample['label_seg'].float().cuda()
            target_data = batch_sample['target_data'].float().cuda()

            seg_out = self.seg_net(target_data, reference_color)

            seg_loss = F.binary_cross_entropy(F.sigmoid(seg_out), label_seg)
            seg_result_ = (F.sigmoid(seg_out).detach() > 0.5)
            mean_iou = get_mean_IOU(seg_result_, label_seg)

            torch.cuda.empty_cache()
            loss_list.append([seg_loss.item(), mean_iou.item()])

            if (i+1)%5==0 and epoch%5==0:
                save_list = []
                title_list = []
                title_list.append('reference_color')
                title_list.append('target_data')
                title_list.append('label_seg')
                title_list.append('seg_result')

                save_list.append(reference_color.detach().to('cpu'))
                save_list.append(target_data.detach().to('cpu'))

                save_list.append(self.__to_color(label_seg))
                save_list.append(self.__to_color(seg_result_))
                save_picture(*save_list, title_list=title_list,
                             path=os.path.join(self.Out_path_val, str(epoch) + '_' + str(i) + '.bmp'),
                             nrow=int(save_list[0].size(0)))
        loss_value = np.mean(np.array(loss_list), axis=0)
        loss_name = ['seg_loss', 'mean_iou']
        print(
            "[TEST][{}/{}],   seg_loss={:.7f}, mean_iou={:.7f}, time={:.7f}".format(i, len(test_loader),
                                                                                     loss_value[0], loss_value[1],
                                                                                     time.time() - start_time))

        return loss_value, loss_name

    def __to_color(self, seg_result):
        '''
        Coloring the results of SegNet
        '''
        images = []
        for i in range(self.batch_size):
            image = np.zeros(shape=(256, 256, 3) ,dtype=np.float)
            for j in range(7):
                image = apply_stroke(image, seg_result[i, j].detach().to('cpu').numpy()>0.5, seg_colors[j])
            images.append(image.transpose((2,0,1)))
        return torch.from_numpy(np.array(images))


if __name__ == '__main__':
    model = TrainSegNet(save_path=os.path.join('out/SegNet_CCSEDB'))
    model.train_model(epochs=10, init_learning_rate=0.0001, batch_size=8, dataset_path=r'dataset_forSegNet_ExtractNet_CCSEDB')

