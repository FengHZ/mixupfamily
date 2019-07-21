import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import random
import numpy as np


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


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


class _DenseLayer2D(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer2D, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition2D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition2D, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock2D(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock2D, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer2D(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet2d(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, num_input_channels=3, growth_rate=32, block_config=(6, 12, 24, 16), compression=0.5,
                 num_init_features=16, bn_size=4, drop_rate=float(0), efficient=False, num_classes=10,
                 data_parallel=True, small_input=True):
        super(DenseNet2d, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        # First convolution
        self.encoder = nn.Sequential()
        self._block_config = block_config
        pre_process = _PreProcess(num_input_channels, num_init_features, small_input=small_input)
        if data_parallel:
            pre_process = nn.DataParallel(pre_process)
        self.encoder.add_module("pre_process", pre_process)
        # add dense block
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock2D(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            if data_parallel:
                block = nn.DataParallel(block)
            self.encoder.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition2D(num_input_features=num_features,
                                      num_output_features=int(num_features * compression))
                if data_parallel:
                    trans = nn.DataParallel(trans)
                self.encoder.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
            else:
                trans = nn.Sequential()
                trans.add_module("norm", nn.BatchNorm2d(num_features))
                trans.add_module("relu", nn.ReLU(inplace=True))
                if data_parallel:
                    trans = nn.DataParallel(trans)
                self.encoder.add_module('transition%d' % (i + 1), trans)
        global_avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if data_parallel:
            global_avg = nn.DataParallel(global_avg)
        self.global_avg = global_avg
        classification = nn.Sequential()
        classification.add_module("fc", nn.Linear(num_features, num_classes))
        if data_parallel:
            classification = nn.DataParallel(classification)
        self.classification = classification
        # Initialization (by xavier)
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                # N_in = param.size(0)*kernel_size
                # N_out= param.size(1)*kernel_size
                # variance = 1/N = 1/(N_in + N_out)/2 = 2/(N_in +N_out)
                nn.init.xavier_uniform_(param.data)
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, input_img, mixup_alpha=None, label=None, manifold_mixup=False, mixup_layer_list=None):
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
                features = getattr(self.encoder, "denseblock%d" % i)(features)
                features = getattr(self.encoder, "transition%d" % i)(features)
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


densenet_dict = {
    "densenet121": {"growth_rate": 32, "block_config": (6, 12, 24, 16), "num_init_features": 64},
    "densenet161": {"growth_rate": 48, "block_config": (6, 12, 36, 24), "num_init_features": 96},
    "densenet169": {"growth_rate": 32, "block_config": (6, 12, 32, 32), "num_init_features": 64},
    "densenet201": {"growth_rate": 32, "block_config": (6, 12, 48, 32), "num_init_features": 64},
    "densenetbc100": {"growth_rate": 12, "block_config": (16, 16, 16), "num_init_features": 24},
    "densenetbc250": {"growth_rate": 24, "block_config": (41, 41, 41), "num_init_features": 48},
    "densenetbc190": {"growth_rate": 40, "block_config": (31, 31, 31), "num_init_features": 40}
}


def get_densenet(name, drop_rate, data_parallel, num_classes):
    return DenseNet2d(growth_rate=densenet_dict[name]["growth_rate"], block_config=densenet_dict[name]["block_config"],
                      num_init_features=densenet_dict[name]["num_init_features"],
                      drop_rate=drop_rate, data_parallel=data_parallel, num_classes=num_classes)
