import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np


class ExtractNet(nn.Module):
    '''
    ExtractNet:extract every single stroke images of the target one by one
               according to the order of the reference strokes.
    '''
    def __init__(self, feature_depth=32):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(4, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True),
                                   nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True))

        self.conv1_2 = nn.Sequential(nn.Conv2d(4, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True),
                                   nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True))

        self.conv2 = nn.Sequential(nn.Conv2d(32+32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True),
                                   nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True),
                                   nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True),
                                   nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))


        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(256, 128, stride=2, kernel_size=4, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True),
                                     nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))  # 64

        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(256, 64, stride=2, kernel_size=4, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True),
                                     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True))  # 128

        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(128, 32, stride=2, kernel_size=4, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True),
                                     nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True))  # 64





        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, 16, stride=2, kernel_size=4, padding=1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True),
                                   nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True))  # 128

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(16, 8, stride=2, kernel_size=4, padding=1),
                                     nn.BatchNorm2d(8),
                                     nn.ReLU(True),
                                     nn.Conv2d(8, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                     #nn.Sigmoid()
                                     )

        self.stn_block = stn(32)


    def forward(self, trans_single, kaiti_seg_tran, seg_out, style_original, feature_64):

        x1 = torch.cat([seg_out, style_original], dim=1)  # 1+3
        x2 = torch.cat([trans_single, kaiti_seg_tran], dim=1)  # 3+1

        out1 = self.conv1(x1)    #  size=64, channel=32
        out2 = self.conv1_2(x2)  #  size=64, channel=32

        out2 = self.stn_block(torch.cat([out1, out2], dim=1), out2)

        out = torch.cat([out1, out2], dim=1)
        out_64 = out
        out = torch.cat([out, feature_64], dim=1) # size=64, channel=48

        out = self.conv2(out)# size=32, channel=64
        out_32 = out
        out = self.conv3(out)# size=16, channel=128
        out_16 = out
        out = self.conv4(out)# size=8, channel=256

        out = self.deconv5(out)  # 16,channel=128
        out = self.deconv4(torch.cat([out, out_16], dim=1))  # 32,channel=64
        out = self.deconv3(torch.cat([out, out_32], dim=1))  # 64,channel=32
        out = self.deconv2(torch.cat([out, out_64], dim=1))  # 128,channel=16

        out = self.deconv1(out)

        return out


class stn(nn.Module):
    def __init__(self,  input_channel):
        super(stn, self).__init__()
        # Spatial transformer localization-network
        self.input_channel = input_channel

        self.localization = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=(5, 5)),  # 60
            nn.MaxPool2d(2, stride=2),  # 30
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=(3, 3)),  # 28
            nn.MaxPool2d(2, stride=2),  # 14
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=(3, 3)),  # 12
            nn.MaxPool2d(2, stride=2),  # 6
            nn.ReLU(True)

        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 3*2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor(np.array([1, 0, 0, 0, 1, 0]), dtype=torch.float))

    def forward(self, x, y):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 6 * 6)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid_ = F.affine_grid(theta, y.size())
        y = F.grid_sample(y, grid_)
        return y

