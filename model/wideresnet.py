import torch
import torch.nn as nn
import numpy as np
import random


class _PreProcess(nn.Sequential):
    def __init__(self, num_input_channels, num_init_features=16, small_input=True):
        super(_PreProcess, self).__init__()
        if small_input:
            self.add_module('conv0',
                            nn.Conv2d(num_input_channels, num_init_features, kernel_size=3, stride=1, padding=1,
                                      bias=True))
        else:
            self.add_module('conv0',
                            nn.Conv2d(num_input_channels, num_init_features, kernel_size=7, stride=2, padding=3,
                                      bias=True))
            self.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                  ceil_mode=False))


class _WideResUnit(nn.Module):
    def __init__(self, num_input_features, num_output_features, stride=1, drop_rate=0.3):
        super(_WideResUnit, self).__init__()
        self.f_block = nn.Sequential()
        self.f_block.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.f_block.add_module('relu1', nn.LeakyReLU(inplace=True))
        self.f_block.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                                   kernel_size=3, stride=stride, padding=1, bias=False))
        self.f_block.add_module('dropout', nn.Dropout(drop_rate))
        self.f_block.add_module('norm2', nn.BatchNorm2d(num_output_features))
        self.f_block.add_module('relu2', nn.LeakyReLU(inplace=True))
        self.f_block.add_module('conv2', nn.Conv2d(num_output_features, num_output_features,
                                                   kernel_size=3, stride=1, padding=1, bias=False))

        if num_input_features != num_output_features or stride != 1:
            self.i_block = nn.Sequential()
            self.i_block.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.i_block.add_module('relu', nn.LeakyReLU(inplace=True))
            self.i_block.add_module('conv',
                                    nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=stride,
                                              bias=False))

    def forward(self, x):
        new_features = self.f_block(x)
        if hasattr(self, "i_block"):
            x = self.i_block(x)
        return new_features + x


class _WideBlock(nn.Module):
    def __init__(self, input_channel, channel_width, block_depth, down_sample=False, drop_rate=0.0):
        super(_WideBlock, self).__init__()
        self.wide_block = nn.Sequential()
        for i in range(block_depth):
            if i == 0:
                unit = _WideResUnit(input_channel, channel_width, stride=int(1 + down_sample),
                                    drop_rate=drop_rate)
            else:
                unit = _WideResUnit(channel_width, channel_width, drop_rate=drop_rate)
            self.wide_block.add_module("wideunit%d" % (i + 1), unit)

    def forward(self, x):
        return self.wide_block(x)


class WideResNet(nn.Module):
    def __init__(self, num_input_channels=3, num_init_features=16, img_size=(32, 32), depth=28, width=2, num_classes=10,
                 data_parallel=True, small_input=True, drop_rate=0.0):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        block_depth = (depth - 4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]
        self._img_size = list(img_size)
        self._widths = widths
        self.encoder = nn.Sequential()
        self.global_avg = nn.Sequential()
        pre_process = _PreProcess(num_input_channels, num_init_features, small_input=small_input)
        if data_parallel:
            pre_process = nn.DataParallel(pre_process)
        self.encoder.add_module("pre_process", pre_process)
        for idx, width in enumerate(widths):
            if idx == 0:
                wide_block = _WideBlock(num_init_features, width, block_depth, drop_rate=drop_rate)
            else:
                wide_block = _WideBlock(widths[idx - 1], width, block_depth, down_sample=True, drop_rate=drop_rate)
            if data_parallel:
                wide_block = nn.DataParallel(wide_block)
            self.encoder.add_module("wideblock%d" % (idx + 1), wide_block)
        if small_input:
            self._img_size = [int(s / 4) for s in self._img_size]
        else:
            self._img_size = [int(s / 16) for s in self._img_size]
        global_avg = nn.AvgPool2d(kernel_size=tuple(self._img_size), stride=1, padding=0)
        # we may use norm and relu before the global avg. Standard implementation doesn't use
        # self.global_avg.add_module("norm", nn.BatchNorm2d(widths[-1]))
        # self.global_avg.add_module('relu', nn.LeakyReLU())
        self.global_avg.add_module('avg', global_avg)
        if data_parallel:
            self.global_avg = nn.DataParallel(self.global_avg)
        classification = nn.Sequential()
        classification.add_module("fc", nn.Linear(widths[-1], num_classes))
        # classification.add_module("logsoftmax", nn.LogSoftmax(dim=1))
        if data_parallel:
            classification = nn.DataParallel(classification)
        self.classification = classification

        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.kaiming_uniform_(param.data)
            elif 'conv' in name and 'bias' in name:
                param.data.fill_(0)
            # initialize liner transform
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)
            # initialize the batch norm layer
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, input_img, mixup_alpha=None, label=None, manifold_mixup=False, mixup_layer_list=None):
        batch_size = input_img.size(0)
        if manifold_mixup:
            # here we follow the source code and only mixup the first three layers
            mixup_layer = random.randint(mixup_layer_list[0], mixup_layer_list[1])
            if mixup_layer == 0:
                input_img, label_a, label_b, lam = self.mixup_data(input_img, label, mixup_alpha)
            features = self.encoder.pre_process(input_img)
            for i in range(1, len(self._widths) + 1):
                if mixup_layer == i:
                    features, label_a, label_b, lam = self.mixup_data(features, label, mixup_alpha)
                features = getattr(self.encoder, "wideblock%d" % i)(features)
            avg_features = self.global_avg(features).view(batch_size, -1)
            cls_result = self.classification(avg_features)
            return cls_result, label_a, label_b, lam
        else:
            features = self.encoder(input_img)
            avg_features = self.global_avg(features).view(batch_size, -1)
            cls_result = self.classification(avg_features)
            return cls_result

    @staticmethod
    def mixup_data(image, label, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = image.size(0)
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_image = lam * image + (1 - lam) * image[index, :]
        label_a, label_b = label, label[index]
        return mixed_image, label_a, label_b, lam


if __name__ == "__main__":
    w = WideResNet()
    for p in w.parameters():
        print(p, p.norm(1))
        input()
