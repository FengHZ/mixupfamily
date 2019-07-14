import torch
from torch import nn
import torch.nn.functional as F


# eps = 1e-7


class CrossEntropyLossWithEps(nn.Module):
    def __init__(self, eps=1e-4):
        super(CrossEntropyLossWithEps, self).__init__()
        self._eps = eps

    def forward(self, score, label):
        """
        :param score: N*C cls score(without softmax)
        :param label: N*C ground truth label with one-hot encode
        :return: cross entropy loss
        """
        loss = torch.sum(-1 * torch.log(F.softmax(score, dim=1) + self._eps) * label, dim=1)
        loss = torch.mean(loss)
        return loss




