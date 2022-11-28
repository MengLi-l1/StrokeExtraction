import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SegNet(nn.Module):
    '''
    SegNet: This separates the target character preliminarily with the guidance of prior information.
    '''

    def __init__(self, out_feature=False):
        super().__init__()
        self.out_feature = out_feature

        self.residual_block_256 = nn.Sequential(BasicBlock(3+3, 16, stride=2, groups=1),
                                                      BasicBlock(16, 32, stride=2, groups=1))# 64
        self.residual_block_64 = nn.Sequential(BasicBlock(32, 64, stride=2, groups=1),
                                                     BasicBlock(64, 64, groups=1),
                                                     BasicBlock(64, 64, groups=1))# 32
        self.residual_block_32 = nn.Sequential(BasicBlock(64, 128, stride=2, groups=1),
                                                     BasicBlock(128, 128, groups=1),
                                                     BasicBlock(128, 128, groups=1)) # 16
        self.residual_block_16 = nn.Sequential(BasicBlock(128, 256, stride=2, groups=1),
                                                     BasicBlock(256, 256, groups=1),
                                                     BasicBlock(256, 256, groups=1)) # 8
        self.de_residual_block_16 = DeBasicBlock(256, 128, stride=2, groups=1) # 16
        self.de_residual_block_32 = DeBasicBlock(256, 64, stride=2, groups=1) # 32
        self.de_residual_block_64 = DeBasicBlock(128, 32, stride=2, groups=1) # 64
        self.de_residual_block_128 = DeBasicBlock(64, 16, stride=2, groups=1) # 128
        self.de_residual_block_256 = DeBasicBlock(16, 16, stride=2, groups=1) # 128

        self.out_conv = nn.Conv2d(16+3, 7, kernel_size=(3, 3), padding=(1, 1))

        self.aspp = ASPP(256, 256)

    def forward(self, target_data, reference_data):
        '''
        @param target_data: shape is (batch, 3, 256, 256)
        @param reference_data: prior information and shape is (batch, 3, 256, 256)
        @return:
        '''
        out_256 = target_data

        out_feature = {}
        input_ = torch.cat([out_256, reference_data], dim=1)# shape=(B, 7*2, 256, 256)
        out_64 = self.residual_block_256(input_) # shape=(B, 7*8, 64, 64)

        out_32 = self.residual_block_64(out_64)    # shape=(B, 14*8, 32, 32)

        out_16 = self.residual_block_32(out_32)    # shape=(B, 14*32, 16, 16)

        out = self.residual_block_16(out_16)    # shape=(B, 14*64, 8, 8)

        out = self.aspp(out)
        out = self.de_residual_block_16(out)  # shape=(B, 14*32, 16, 16)
        out = torch.cat([out, out_16], dim=1)
        out = self.de_residual_block_32(out) # shape=(B, 14*16, 32, 32)
        out = torch.cat([out, out_32], dim=1)
        out = self.de_residual_block_64(out)  # shape=(B, 14*8, 64, 64)
        out_feature['out_64_32'] = out
        out = torch.cat([out, out_64], dim=1)
        out = self.de_residual_block_128(out)  # shape=(B, 14*4, 128, 128)
        out_feature['out_128_16'] = out
        out = self.de_residual_block_256(out)  # shape=(B, 14*4, 128, 128)
        out = self.out_conv(torch.cat([out, out_256], dim=1))  # shape=(B, 7, 256, 256)
        if self.out_feature:
            return out, out_feature
        else:
            return out


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 3, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DeBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups:int=1,

    ) -> None:
        super(DeBasicBlock, self).__init__()



        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1_groups = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=1),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(True))

        self.conv2_groups = nn.Sequential(nn.ConvTranspose2d(planes, planes, stride=2, kernel_size=4, padding=1),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(True)
                                          )
        self.stride = stride

    def forward(self, x):
        out_g = self.conv1_groups(x)
        out_g = self.conv2_groups(out_g)
        return out_g


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups:int=1,
        base_width: int = 64,
        dilation: int = 1,

    ) -> None:
        super(BasicBlock, self).__init__()

        norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=1, groups=groups, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1, groups=groups)
        self.bn2 = norm_layer(planes)
        self.downsample=None
        if inplanes != planes or stride!=1:
            self.downsample = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=stride)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


