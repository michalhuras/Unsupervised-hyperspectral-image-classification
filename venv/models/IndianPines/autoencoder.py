#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from scipy import io
import numpy as np
import mathematical_operations as mo


class AutoencoderNone(nn.Module):
    def __init__(self):
        super(AutoencoderNone, self).__init__()

        self.my_tensor = torch.Tensor()

    def parameters(self):
        return self.my_tensor

    def forward(self, x):
        return x

    def getCode(self, x):
        return x

    def getType(self):
        return 'none'

    def getName(self):
        return 'none'


class AutoencoderLinear(nn.Module):
    def __init__(self):
        super(AutoencoderLinear, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(200, 90),
            nn.Linear(90, 40),
            nn.Linear(40, 20))
        self.decoder = nn.Sequential(
            nn.Linear(20, 40),
            nn.Linear(40, 90),
            nn.Linear(90, 200))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def getCode(self, x):
        x = self.encoder(x)
        return x

    def getType(self):
        return 'linear'

    def getName(self):
        return '1'
