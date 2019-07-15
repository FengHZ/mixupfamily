import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class _PreProcess(nn.Sequential):
    def __init__(self, num_input_channels, num_init_features=64, small_input=True):
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


class _PreActUnit(nn.Module):
    """
    Pre Activation of the Basic Block
    """

    def __init__(self, num_input_features, num_output_features, expansion=1, stride=1, drop_rate=0.0):
        super(_PreActUnit, self).__init__()
        self._expansion = expansion
        self.f_block = nn.Sequential()
        if self._expansion == 1:
            self.f_block.add_module('norm1', nn.BatchNorm2d(num_input_features))
            self.f_block.add_module('relu1', nn.ReLU(inplace=True))
            self.f_block.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                                       kernel_size=3, stride=stride, padding=1, bias=False))
            self.f_block.add_module('dropout', nn.Dropout(drop_rate))
            self.f_block.add_module('norm2', nn.BatchNorm2d(num_output_features))
            self.f_block.add_module('relu2', nn.ReLU(inplace=True))
            self.f_block.add_module('conv2', nn.Conv2d(num_output_features, num_output_features,
                                                       kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.f_block.add_module('norm1', nn.BatchNorm2d(num_input_features))
            self.f_block.add_module('relu1', nn.ReLU(inplace=True))
            self.f_block.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                                       kernel_size=1, bias=False))
            self.f_block.add_module('norm2', nn.BatchNorm2d(num_output_features))
            self.f_block.add_module('relu2', nn.ReLU(inplace=True))
            self.f_block.add_module('conv2', nn.Conv2d(num_output_features, num_output_features,
                                                       kernel_size=3, stride=stride, padding=1, bias=False))
            self.f_block.add_module('dropout', nn.Dropout(drop_rate))
            self.f_block.add_module('norm3', nn.BatchNorm2d(num_output_features))
            self.f_block.add_module('relu3', nn.ReLU(inplace=True))
            self.f_block.add_module('conv3', nn.Conv2d(num_output_features, self._expansion * num_output_features,
                                                       kernel_size=1, bias=False))
        if stride != 1 or num_input_features != self._expansion * num_output_features:
            self.i_block = nn.Sequential()
            self.i_block.add_module('norm', nn.BatchNorm2d(num_input_features))
            # self.i_block.add_module('relu', nn.ReLU(inplace=True))
            self.i_block.add_module('conv',
                                    nn.Conv2d(num_input_features, self._expansion * num_output_features, kernel_size=1,
                                              stride=stride,
                                              bias=False))

    def forward(self, x):
        new_features = self.f_block(x)
        if hasattr(self, "i_block"):
            x = self.i_block(x)
        return new_features + x


class _PreActBlock(nn.Module):
    def __init__(self, input_channel, output_channel, expansion, block_depth, down_sample=False, drop_rate=0.0):
        super(_PreActBlock, self).__init__()
        self.preact_block = nn.Sequential()
        for i in range(block_depth):
            if i == 0:
                unit = _PreActUnit(input_channel, output_channel, expansion, stride=int(1 + down_sample),
                                   drop_rate=drop_rate)
            else:
                unit = _PreActUnit(input_channel, output_channel, expansion, drop_rate=drop_rate)
            self.preact_block.add_module("unit%d" % (i + 1), unit)
            input_channel = output_channel * expansion

    def forward(self, x):
        return self.preact_block(x)


class PreActResNet(nn.Module):
    def __init__(self, expansion, block_config, num_input_channels=3, num_init_features=64, img_size=(32, 32),
                 num_classes=10, data_parallel=True, small_input=True, drop_rate=0.0):
        super(PreActResNet, self).__init__()
        self._img_size = list(img_size)
        self._input_channels = num_init_features
        self._output_channels = num_init_features
        self.encoder = nn.Sequential()
        self.global_avg = nn.Sequential()
        self._block_config = block_config
        pre_process = _PreProcess(num_input_channels, num_init_features, small_input=small_input)
        if data_parallel:
            pre_process = nn.DataParallel(pre_process)
        self.encoder.add_module("pre_process", pre_process)
        for idx, block_depth in enumerate(block_config):
            block = _PreActBlock(self._input_channels, self._output_channels, expansion, block_depth,
                                 down_sample=(idx != 0), drop_rate=drop_rate)
            # update the channel num
            self._input_channels = self._output_channels * expansion
            self._output_channels = self._output_channels * 2
            if data_parallel:
                block = nn.DataParallel(block)
            self.encoder.add_module("block%d" % (idx + 1), block)
        if small_input:
            factor = 2 ** (len(block_config) - 1)
            self._img_size = [int(s / factor) for s in self._img_size]
        else:
            factor = 4 * (2 ** (len(block_config) - 1))
            self._img_size = [int(s / factor) for s in self._img_size]
        global_avg = nn.AvgPool2d(kernel_size=tuple(self._img_size), stride=1, padding=0)
        # we may use norm and relu before the global avg. Standard implementation doesn't use
        # self.global_avg.add_module("norm", int(num_init_features * (2 ** (len(block_config) - 1)) * expansion))
        # self.global_avg.add_module('relu', nn.LeakyReLU())
        self.global_avg.add_module('avg', global_avg)
        if data_parallel:
            self.global_avg = nn.DataParallel(self.global_avg)
        classification = nn.Sequential()
        classification.add_module("fc", nn.Linear(num_init_features * (2 ** (len(block_config) - 1)) * expansion,
                                                  num_classes))
        if data_parallel:
            classification = nn.DataParallel(classification)
        self.classification = classification

        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            elif 'conv' in name and 'bias' in name:
                param.data.fill_(0)
            # initialize liner transform
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)
            # initialize the batch norm layer
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, input_img, mixup_alpha=None, label=None, manifold_mixup=False,mixup_layer_list=None):
        batch_size = input_img.size(0)
        if manifold_mixup:
            # here we follow the source code and only mixup the first three layers
            mixup_layer = random.randint(mixup_layer_list[0], mixup_layer_list[1])
            if mixup_layer == 0:
                input_img, label_a, label_b, lam = self.mixup_data(input_img, label, mixup_alpha)
            features = self.encoder.pre_process(input_img)
            for i in range(1, len(self._block_config) + 1):
                if mixup_layer == i:
                    features, label_a, label_b, lam = self.mixup_data(features, label, mixup_alpha)
                features = getattr(self.encoder, "block%d" % i)(features)
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


preactresnet_dict = {
    "preactresnet18": {"expansion": 1, "block_config": [2, 2, 2, 2]},
    "preactresnet34": {"expansion": 1, "block_config": [3, 4, 6, 3]},
    "preactresnet50": {"expansion": 4, "block_config": [3, 4, 6, 3]},
    "preactresnet101": {"expansion": 4, "block_config": [3, 4, 23, 3]},
    "preactresnet152": {"expansion": 4, "block_config": [3, 8, 36, 3]}
}


def get_preact_resnet(name, img_size, drop_rate, data_parallel, num_classes):
    return PreActResNet(expansion=preactresnet_dict[name]["expansion"],
                        block_config=preactresnet_dict[name]["block_config"], img_size=img_size, drop_rate=drop_rate,
                        data_parallel=data_parallel, num_classes=num_classes)
