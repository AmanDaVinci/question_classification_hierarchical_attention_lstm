import torch
import torch.nn as nn
import numpy as np


class Highway(nn.Module):
    def __init__(self, in_features, out_features, layer_size=1, bias=-2, g=nn.ReLU()):
        # TODO: Not sure if in_features = out_features
        # TODO: Not sure of purpose of layer_size and bias arguments
        self.t = nn.Linear(in_features, out_features)
        self.sigm = nn.Sigmoid()
        self.g = nn.Sequential(
            nn.Linear(in_features, out_features),
            g
        )
        self.bias = bias

    def forward(self, input):
        transform_gate = self.sigm(self.t(input) + self.bias)
        carry_gate = 1 - transform_gate
        output = self.g()
        output = transform_gate * output + carry_gate * input
        return output
