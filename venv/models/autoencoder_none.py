#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from scipy import io
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, unused_parameter):
        super(Autoencoder, self).__init__()

        self.my_tensor = torch.Tensor()

    def parameters(self):
        return self.my_tensor

    def forward(self, x):
        return x

    def getCode(self, x):
        return x

    @staticmethod
    def getType():
        return 'none'

    @staticmethod
    def getName():
        return 'none'
