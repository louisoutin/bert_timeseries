import torch
from torch import nn


class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"

    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class GACP1d(nn.Module):
    "Global AdaptiveConcatPool + Flatten"

    def __init__(self, output_size=1):
        super().__init__()
        self.gacp = AdaptiveConcatPool1d(output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.gacp(x))


class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"

    def __init__(self, output_size=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.gap(x))


class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"

    def __init__(self, n_in, n_out, bn=True, p=0.0, act=None, lin_first=False):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None:
            lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low


class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        super().__init__()

    def forward(self, x):
        return sigmoid_range(x, self.low, self.high)
