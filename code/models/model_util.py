"""
Model building utils
"""


import torch
import torch.nn as nn
import math


def new_parameter(size):
    p = torch.zeros(size)
    stdv = 1.0 / math.sqrt(size)
    p.data.uniform_(-stdv, stdv)
    return nn.Parameter(p)


def reset_sequential(seq):
    for layer in seq:
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()
