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
    def __init__(self, first_size):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(first_size, 90),
            nn.ReLU(True),
            nn.Linear(90, 40),
            nn.ReLU(True),
            nn.Linear(40, 15),
            nn.ReLU(True),
            nn.Linear(15, 7),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(7, 15),
            nn.ReLU(True),
            nn.Linear(15, 40),
            nn.ReLU(True),
            nn.Linear(40, 90),
            nn.ReLU(True),
            nn.Linear(90, first_size),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def getCode(self, x):
        x = self.encoder(x)
        return x

    @staticmethod
    def getType():
        return 'linear'

    @staticmethod
    def getName():
        return '6'
