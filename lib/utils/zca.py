import torch


def apply_zca(data, zca_mean, zca_components):
    """
    :param data: torch.tensor with B*C*H*W
    :param zca_mean: 1*3072 cuda tensor
    :param zca_components: 3072*3072 cuda tensor
    :return: zca normalized data
    """
    B, C, H, W = data.size()
    data = data.view(B, -1)
    data = torch.mm(data - zca_mean, torch.t(zca_components))
    data = data.view(B, C, H, W)
    return data
