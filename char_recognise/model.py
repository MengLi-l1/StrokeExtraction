import torch
import torch.nn as nn
'''
A Recognise model based on VGGNet for Chinese characters.
'out_vgg_bn' is result of training
'''

class CharRecognise(nn.Module):

    def __init__(self, num_classes: int, out_feature=False, init_weights: bool = True):
        super(CharRecognise, self).__init__()
        self.out_feature = out_feature

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True)) # 64
        
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))  # 32

        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))  # 16
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))  # 8

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),

        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = {}
        x = self.conv1(x)
        feature['out_64_16'] = x
        x = self.conv2(x)
        feature['out_32_32'] = x
        x = self.conv3(x)
        feature['out_16_64'] = x
        x = self.conv4(x)    
        feature['out_8_128'] = x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.out_feature:
            return x, feature
        else:
            return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
