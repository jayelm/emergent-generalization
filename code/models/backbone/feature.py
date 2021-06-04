"""
Feature processing backbones
"""


import torch.nn as nn
from .. import model_util


class FeatureMLP(nn.Module):
    def __init__(self, input_size=16, output_size=16, n_layers=2):
        super().__init__()
        assert n_layers >= 2, "Need at least 2 layers"

        layers = [nn.Linear(input_size, output_size)]

        for _ in range(n_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(output_size, output_size))

        self.trunk = nn.Sequential(*layers)
        self.n_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size
        self.final_feat_dim = self.output_size

    def forward(self, x):
        return self.trunk(x)

    def reset_parameters(self):
        model_util.reset_sequential(self.trunk)
