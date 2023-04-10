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
            nn.Linear(156, 75),
            nn.Linear(75, 35),
            nn.Linear(35, 15))
        self.decoder = nn.Sequential(
            nn.Linear(15, 35),
            nn.Linear(35, 75),
            nn.Linear(75, 156))

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
