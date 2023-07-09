import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import os
from char_recognise.model import CharRecognise
import numpy as np
char_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../char_recognise/out_vgg_bn/model/model.th')


#######################################################################
#  SDNet
#######################################################################
class SDNet(nn.Module):
    """
    SDNet:Deep Structure Deformable Image Registration
    """

    def __init__(self, image_size=(256, 256)):
        super().__init__()
        self.unet_model = UNetWithFeature(with_char_feature=True)

        nd = Normal(0, 1e-5)
        self.flow_whole = nn.Conv2d(8, 2, kernel_size=3, padding=1)
        self.flow_whole.weight = nn.Parameter(nd.sample(self.flow_whole.weight.shape))
        self.flow_whole.bias = nn.Parameter(torch.zeros(self.flow_whole.bias.shape))
        self.flow_linear = nn.Conv2d(8, 2, kernel_size=3, padding=1)
        self.flow_linear.weight = nn.Parameter(nd.sample(self.flow_linear.weight.shape))
        self.flow_linear.bias = nn.Parameter(torch.zeros(self.flow_linear.bias.shape))

        self.spatial_transform = SpatialTransformer(image_size)

        # coordinate matrices
        vectors = [torch.arange(0, s) for s in [256, 256]]
        coordinate = torch.meshgrid(vectors)
        coordinate = torch.stack(coordinate)[[1, 0]].float()  # y, x 调整为x,y
        self.coordinate = coordinate.numpy()

    def get_two_registration_field(self, color_reference, target_data):
        '''
        Calculate Registration field of Φ_d and Φ_s
        '''
        x_whole, x_linear = self.unet_model(color_reference, target_data)
        flow_global = self.flow_whole(x_whole) * 10   # Registration field of Φ_d
        flow_linear_ = self.flow_linear(x_linear)*5
        flow_linear = flow_global + flow_linear_  # Registration field of Φ_s
        transformed_target_data, _ = self.spatial_transform(target_data, flow_global)
        _, grid_for_linear = self.spatial_transform(target_data, flow_linear)
        return transformed_target_data, flow_global, grid_for_linear

    def __get_inverse_grid(self, grid):
        '''
        Get inverse gird
        @param grid: shape is (1, 2, 256, 256)
        @return:
        '''
        grid_ = (grid.detach().to('cpu').numpy()+1)*127.5
        x = np.array([grid_[0, :, 0, 0], grid_[0, :, 0, 255], grid_[0, :, 255, 0]], dtype=np.float)
        y = np.array([[0, 0], [255, 0], [0, 255]], dtype=np.float)
        expand = np.ones(shape=(3, 1), dtype=np.float)
        x = np.concatenate([x, expand], axis=1)
        map = np.dot(np.linalg.inv(x), y)
        grid_base = self.coordinate.reshape((2, -1)).transpose((1, 0)).astype(np.float)
        grid_base = np.concatenate([grid_base, np.ones(shape=(256*256, 1), dtype=np.float)], axis=1)
        d_grid = np.round(np.dot(grid_base, map)).astype(np.int)
        d_grid = np.clip(d_grid[:, ], 0, 255).reshape((1, 256, 256, 2)).transpose((0, 3, 1, 2 )).astype(np.float)
        d_grid = d_grid / 127.5 - 1
        return torch.from_numpy(d_grid).float().cuda()

    def get_linear_estimation(self, reference_single_stroke, grids, reference_single_stroke_centroid, inverse=False):
        '''
        Calculate Linear Estimation of Single Stroke Spatial Transformation
        @param reference_single_stroke: tensor, shape=(N, 1, 256, 256)
        @param grids: tensor, shape=(N, 256, 256, 2)
        @param refer_image: tensor, shape=(N, 2)
        @return: Liner grid
        '''
        grid_ = torch.transpose(grids, 2, 3)
        grid_ = torch.transpose(grid_, 1, 2)
        mark = (reference_single_stroke > 0.5).float().cuda()  # （N, 1, 256, 256)

        # Mean value of local marked region
        mean_xy = torch.sum(mark * grid_, dim=[2, 3], keepdim=True) / (
                    torch.sum(mark, dim=[2, 3], keepdim=True) + 0.0001)

        # Centroid of single reference stroke image
        # X-Px; Y-Py
        center_refer = torch.reshape(reference_single_stroke_centroid, (-1, 2, 1, 1))
        grid = torch.from_numpy(self.coordinate).cuda().float().unsqueeze(0)
        grid = grid.repeat(center_refer.size(0), 1, 1, 1)
        center = center_refer
        grid -= center  # (N,2,256,256)

        # Mean value of the first derivative
        x_drive = torch.zeros(size=(1, 1, 1, 5)).float().cuda()
        x_drive[0, 0, 0, 0] = -1
        x_drive[0, 0, 0, 4] = 1
        x_drive /= 4
        x_drive = x_drive.repeat((2, 1, 1, 1))
        y_drive = torch.zeros(size=(1, 1, 5, 1)).float().cuda()
        y_drive[0, 0, 0, 0] = -1
        y_drive[0, 0, 4, 0] = 1
        y_drive /= 4
        y_drive = y_drive.repeat((2, 1, 1, 1))
        dx = F.conv2d(grid_, x_drive, padding=(0, 2), groups=2)
        dy = F.conv2d(grid_, y_drive, padding=(2, 0), groups=2)
        dx_mean = torch.sum(dx * mark, dim=[2, 3], keepdim=True) / (torch.sum(mark, dim=[2, 3], keepdim=True) + 0.0001)
        dy_mean = torch.sum(dy * mark, dim=[2, 3], keepdim=True) / (torch.sum(mark, dim=[2, 3], keepdim=True) + 0.0001)

        # Calculate Linear Estimation of Single Stroke Spatial Transformation
        dx_value = dx_mean * grid[:, :1]
        dy_value = dy_mean * grid[:, 1:]
        linear_grid_ = mean_xy + dx_value + dy_value

        # if in the model inference, get the inverse grid
        if inverse:
            linear_grid = self.__get_inverse_grid(linear_grid_)
        else:
            linear_grid = linear_grid_

        linear_grid = torch.transpose(linear_grid, 1, 2)
        linear_grid = torch.transpose(linear_grid, 2, 3)
        return linear_grid


class inception_block(nn.Module):
    """
    A inception convolution block used in first layer of main frame
    """

    def __init__(self, in_channels, out_channels, stride=1, group=1):
        super().__init__()
        self.b2_3x3 = nn.Sequential(nn.Conv2d(in_channels, int(out_channels/2), stride=stride, kernel_size=3, padding=1, groups=group),
                                    nn.BatchNorm2d(int(out_channels/2)),
                                    nn.LeakyReLU(0.2))
        self.b2_7x7 = nn.Sequential(nn.Conv2d(in_channels, int(out_channels/2), stride=stride, kernel_size=3, padding=3, dilation=3, groups=group),
                                    nn.BatchNorm2d(int(out_channels/2)),
                                    nn.LeakyReLU(0.2))
    def forward(self, x):
        out_3x3 = self.b2_3x3(x)
        out_7x7 = self.b2_7x7(x)

        return torch.cat([out_3x3, out_7x7], dim=1)


class conv_block(nn.Module):
    """
    A convolution block used in main frame
    """
    def __init__(self, in_channels, out_channels, stride=1, group=1):

        super(conv_block, self).__init__()
        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, 1, groups=group),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU(0.2),
                                  )
    def forward(self, x):
        out = self.main(x)
        return out


class SpatialTransformer(nn.Module):
    """
    uses the output from the UNet to bulid an grid_sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)[[1, 0]]
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode


    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
        return F.grid_sample(src, new_locs, mode=self.mode), new_locs


class UNetWithFeature(nn.Module):
    '''
    The main frame of SDNet similar to UNet
    '''
    def __init__(self,  with_char_feature=True):
        super().__init__()
        self.with_char_feature = with_char_feature
        self._norm_layer = nn.BatchNorm2d
        self.conv_input = nn.Sequential(inception_block(4, 16, stride=2, group=1))  # 128
        self.block0 = conv_block(16, 32, stride=2, group=1)  # 64
        self.block1 = conv_block(32, 64, stride=2, group=1)  # 32
        self.block2 = conv_block(64, 128, stride=2, group=1)  # 16
        self.block3 = conv_block(128, 256, stride=2, group=1)  # 8

        self.de_block1 = conv_block(256, 128, group=1)  # 16
        self.de_block2 = conv_block(128 + 128, 64, group=1)  # up  # 32
        self.de_block3 = conv_block(64+64, 32, group=1)  # up # 64
        self.de_block4 = conv_block(32+32, 16, group=1)  # up  # 128
        self.de_block5 = conv_block(16+16, 8, group=1)  # up  # 256

        self.de_block6 = conv_block(8, 2)   # 256

        # ConvLayer for fusion Features of Chinese Character Recognise Net
        self.conv_fusion1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                                          nn.BatchNorm2d(32),
                                          nn.LeakyReLU(0.2, inplace=True))
        self.conv_fusion2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.conv_fusion3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.conv_fusion4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3, 3), padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.2, inplace=True))

        # Upsampling operation and The second branch of Registration Field
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.second_brach = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'),
                                          nn.Conv2d(32, 8, kernel_size=(9, 9), padding=4),
                                          nn.BatchNorm2d(8),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Upsample(scale_factor=2, mode='bilinear')
                                          )

        # Chinese Character Recognise Net
        if with_char_feature:
            self.char_net = CharRecognise(num_classes=3500)
            self.char_net.out_feature = True

            if os.path.exists(char_model_path):
                state = torch.load(char_model_path)
                self.char_net.load_state_dict(state['state'])
                print("Successful to load parameters of CharNet in SDNet!")
            else:
                print("Failed to load parameters of CharNet in SDNet!")
            self.char_net.eval()  # 固定BN参数
            self.char_net.requires_grad_(False)  # 停止更新参数

    def train(self, mode=True):
        '''
        overload train function
        '''
        super().train(mode)
        if self.with_char_feature:
            self.char_net.eval()

    def requires_grad_(self, requires_grad: bool = True):
        '''
        overload requires_grad_ function
        '''
        super().requires_grad_(requires_grad)
        if self.with_char_feature:
            self.char_net.requires_grad_(False)

    def forward(self, color_reference, target_image):
        reference_original = torch.sum(color_reference, keepdim=True, dim=1)>0.1
        char_x = F.interpolate(reference_original.float(), size=(128, 128))
        _, char_feature_reference = self.char_net(char_x)
        char_x = F.interpolate(target_image, size=(128, 128))
        _, char_feature_target = self.char_net(char_x)

        feature = {}

        x = self.conv_input(torch.cat([color_reference, target_image], dim=1))
        feature['out_128'] = x    # channel=16
        x = self.block0(x)  # channel=32, size=64
        x = torch.cat([x, char_feature_reference['out_64_16'], char_feature_target['out_64_16']], dim=1)
        x = self.conv_fusion1(x)
        feature['out_64'] = x      # channel=32

        # 32
        x = self.block1(x)  # 64
        x = torch.cat([x, char_feature_reference['out_32_32'], char_feature_target['out_32_32']], dim=1)
        x = self.conv_fusion2(x)
        feature['out_32'] = x   # channel=64

        # 16
        x = self.block2(x)  # channel=128
        x = torch.cat([x, char_feature_reference['out_16_64'], char_feature_target['out_16_64']], dim=1)
        x = self.conv_fusion3(x)
        feature['out_16'] = x   # channel=128

        # 8
        x = self.block3(x)  # 8
        x = torch.cat([x, char_feature_reference['out_8_128'], char_feature_target['out_8_128']], dim=1)  # 添加识别特征
        x = self.conv_fusion4(x)  # channel=256

        x = self.de_block1(x)  # 8
        x = self.upsample(x)   # 16  # channel=256
        x = torch.cat([x, feature['out_16']], dim=1)

        x = self.de_block2(x)  # 16
        x = self.upsample(x)   # 32

        x = torch.cat([x, feature['out_32']], dim=1)
        x = self.de_block3(x)
        out_32 = x

        second_branch_out = self.second_brach(out_32)


        x = self.upsample(x)      # 64

        x = torch.cat([x, feature['out_64']], dim=1)
        x = self.de_block4(x)
        x = self.upsample(x)    # 128
        x = torch.cat([x, feature['out_128']], dim=1)

        x = self.de_block5(x)
        x = self.upsample(x)   # 256

        return x, second_branch_out
