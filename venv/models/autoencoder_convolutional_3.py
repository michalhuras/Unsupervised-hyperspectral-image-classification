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
        print("First size ", self.first_size)
        self.dimensions = 10
        self.kernel_size = 20
        self.pooling_value = 4
        self.elements_in_dimension = int((first_size - (self.kernel_size - 1)) / self.pooling_value)
        print("Elements ", self.elements_in_dimension)

        # wyliczane w trakcie działania programu
        self.this_size = 0
        self.indices = 0

        self.encoder_conv = nn.Conv1d(1, self.dimensions, self.kernel_size)
        self.encoder_pool = nn.MaxPool1d(self.pooling_value, stride=self.pooling_value, return_indices=True)
        self.encoder_linear = nn.Linear(self.elements_in_dimension, 1)

        self.decoder_linear = nn.Linear(1, self.elements_in_dimension)
        self.decoder_pool = nn.MaxUnpool1d(self.pooling_value, stride=self.pooling_value)
        self.decoder_conv = nn.ConvTranspose1d(self.dimensions, 1, self.kernel_size)

    def encode(self, input_value):
        input_value = input_value.reshape((1, 1, input_value.shape[1]))
        input_value = F.relu(self.encoder_conv(input_value))
        self.this_size = input_value.size()
        input_value, self.indices = self.encoder_pool(input_value)
        input_value = input_value.reshape((input_value.shape[1], input_value.shape[2]))
        input_value = self.encoder_linear(input_value)
        return input_value

    def decode(self, input_value):
        input_value = input_value.reshape((input_value.shape[0], 1))
        input_value = self.decoder_linear(input_value)
        input_value = input_value.reshape((1, input_value.shape[0], input_value.shape[1]))
        input_value = self.decoder_pool(input_value, self.indices, self.this_size)
        input_value = self.decoder_conv(input_value)
        input_value = input_value.reshape((1, self.first_size))
        return input_value

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def getCode(self, x):
        print("Get code")
        x = self.encode(x)
        x = x.reshape((x.shape[0]))
        print("Shape:", x.shape())
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
        return '3'
