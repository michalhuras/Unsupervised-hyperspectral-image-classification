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

        print("First size: ", first_size)

        self.first_size = first_size
        self.dimensions = 6
        self.kernel_size = 9

        self.encoder_conv = nn.Conv1d(1, self.dimensions, self.kernel_size)
        self.encoder_lin = nn.Sequential(
            nn.Linear(self.dimensions * (self.first_size - 8), 500),
            nn.Linear(500, 200),
            nn.Linear(200, 80),
            nn.Linear(80, 30),
            nn.Linear(30, 10))

        self.decoder_lin = nn.Sequential(
            nn.Linear(10, 30),
            nn.Linear(30, 80),
            nn.Linear(80, 200),
            nn.Linear(200, 500),
            nn.Linear(500, self.dimensions * (self.first_size - 8)))
        self.decoder_conv = nn.ConvTranspose1d(self.dimensions, 1, self.kernel_size)

        # 8 ponieważ kernel jest 9 i "ucina" po połowie kernela z każdej ze stron

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape((1, 1, x.shape[1]))
        # print("Encoder")
        x = F.relu(self.encoder_conv(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.encoder_lin(x)
        # print("Decoder")
        x = self.decoder_lin(x)
        x = x.view(-1, 6, self.first_size - 8)
        x = self.decoder_conv(x)
        x = x.reshape(original_shape)
        return x

    def getCode(self, x):
        x = x.reshape((1, 1, np.amax(x.shape)))
        # print("Encoder")
        x = F.relu(self.encoder_conv(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.encoder_lin(x)
        x = x.reshape((10))
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @staticmethod
    def getType():
        return 'convolutional'

    @staticmethod
    def getName():
        return '1'
