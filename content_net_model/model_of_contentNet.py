
import torch.nn.functional as F
import torch.nn as nn
import torch


class ContentNet(nn.Module):
    def __init__(self, embedding1_length=512, embedding2_length=256):
        '''
        ContentNet:Obtain content feature of stroke
        @param embedding1_length: Length of deep features
        @param embedding2_length: Length of shallow features
        '''
        super().__init__()
        # conv1: padding=same
        self.conv1 = nn.Sequential(GoogleConv(1, 16, stride=2),
                                   nn.BatchNorm2d(16))  # 128
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=(1, 1), stride=1),
                                     nn.BatchNorm2d(16))  # 128
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=2, stride=2),
                                   nn.BatchNorm2d(32))  # 64
        self.conv2_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=(1, 1), stride=1),
                                     nn.BatchNorm2d(32))  # 64
        self.conv2_embedding = nn.Sequential(nn.Conv2d(32, 2, kernel_size=3, padding=(1, 1), stride=1)) # 64

        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=2, padding=(1, 1), stride=2),
                                   nn.BatchNorm2d(64))  # 32
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1), stride=1),
                                     nn.BatchNorm2d(64))  # 32

        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=2, stride=2),
                                   nn.BatchNorm2d(128))  # 16
        self.conv4_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1), stride=1),
                                     nn.BatchNorm2d(128))  # 16
        self.conv5 = nn.Sequential(nn.Conv2d(128, 32, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(32),
                                   )  # 16

        self.liner6 = nn.Linear(32 * 16 * 16, embedding1_length)  # 深层特征
        self.deliner6 = nn.Linear(embedding1_length, 32 * 16 * 16)

        self.liner7 = nn.Linear(2 * 64 * 64, embedding2_length)  # 浅层特征
        self.deliner7 = nn.Linear(embedding2_length, 2 * 64 * 64)
        self.deconv_7 = nn.Sequential(nn.Conv2d(2, 16, stride=1, kernel_size=1),
                                      nn.BatchNorm2d(16))

        # 7
          # 16
        self.dconv5_2 = nn.Sequential(nn.Conv2d(32, 128, stride=1, kernel_size=1),
                                      nn.BatchNorm2d(128),
                                      nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=(1, 1)),
                                      nn.BatchNorm2d(128))  # 16
        self.dconv4 = nn.Sequential(nn.ConvTranspose2d(128, 64, stride=2, kernel_size=2),
                                    nn.BatchNorm2d(64))  # 32
        self.dconv4_2 = nn.Sequential(nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=(1, 1)),
                                      nn.BatchNorm2d(64))  # 32

        self.dconv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, stride=2, kernel_size=2),
                                    nn.BatchNorm2d(32))  # 64
        self.dconv3_2 = nn.Sequential(nn.Conv2d(48, 32, stride=1, kernel_size=3, padding=(1, 1)),
                                      nn.BatchNorm2d(32))  # 64

        self.dconv2 = nn.Sequential(nn.ConvTranspose2d(32, 16, stride=2, kernel_size=2),
                                    nn.BatchNorm2d(16))  # 128
        self.dconv2_2 = nn.Sequential(nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=(1, 1)),
                                      nn.BatchNorm2d(16))  # 128

        self.dconv1 = nn.Sequential(nn.ConvTranspose2d(16, 8, stride=2, kernel_size=2))  # 256
        self.dconv1_2 = nn.Conv2d(8, 1, stride=1, kernel_size=3, padding=(1, 1))  # 256

    def forward(self, input):
        # Encoder
        # 1. 256 x 256 -> 128 x 128
        x = F.relu(self.conv1(input), True)

        x = F.relu(self.conv1_2(x), True)
        # 1. 128 x 128 -> 64 x 64
        x = F.relu(self.conv2(x), True)
        x = F.relu(self.conv2_2(x), True)
        em_x = F.relu(self.conv2_embedding(x), True)
        # 3. 64 x 64 -> 32 x 32
        x = F.relu(self.conv3(x), True)
        x = F.relu(self.conv3_2(x), True)

        # 4. 32 x 32 -> 16 x 16
        x = F.relu(self.conv4(x), True)
        x = F.relu(self.conv4_2(x), True)

        # # 5. 16 x 16 -> 5 x 5
        x = F.relu(self.conv5(x), True)

        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.liner6(x), True)
        embedding = x
        x = F.relu(self.deliner6(x), True)
        x = x.view(-1, 32, 16, 16)

        em_x = em_x.view(-1, 2 * 64 * 64)
        em_x = F.relu(self.liner7(em_x), True)
        embedding = torch.cat([embedding, em_x], dim=1)
        em_x = F.relu(self.deliner7(em_x), True)
        em_x = em_x.view(-1, 2, 64, 64)
        em_x = F.relu(self.deconv_7(em_x), True)

        # 8. 5 x 5 -> 16 x 16


        x = F.relu(self.dconv5_2(x), True)

        # 9. 16 x 16 -> 32 x 32
        x = F.relu(self.dconv4(x), True)

        x = F.relu(self.dconv4_2(x), True)

        # 10. 32 x 32 -> 64 x 64
        x = F.relu(self.dconv3(x), True)
        x = torch.cat([x, em_x], dim=1)
        x = F.relu(self.dconv3_2(x), True)

        # 11. 64 x 64 -> 128 x 128
        x = F.relu(self.dconv2(x), True)
        x = F.relu(self.dconv2_2(x), True)

        # 12. 128 x 238 -> 256 x 256
        x = F.relu(self.dconv1(x), True)
        x = F.relu(self.dconv1_2(x), True)
        x = F.tanh(x)
        return embedding, x


class GoogleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.b2_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels / 4), stride=stride, kernel_size=1, padding=0),
            nn.BatchNorm2d(int(out_channels / 4))
        )
        self.b2_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels / 4), stride=stride, kernel_size=3, padding=1),
            nn.BatchNorm2d(int(out_channels / 4))
            )
        self.b2_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels / 4), stride=stride, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(int(out_channels / 4))

            )
        self.b2_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels / 4), stride=stride, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(int(out_channels / 4))
            )
    def forward(self, x):
        out_1x1 = self.b2_1x1(x)
        out_3x3 = self.b2_3x3(x)
        out_5x5 = self.b2_5x5(x)
        out_7x7 = self.b2_7x7(x)

        return torch.cat([out_1x1, out_3x3, out_5x5, out_7x7], dim=1)

