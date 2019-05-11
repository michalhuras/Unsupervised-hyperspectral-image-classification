#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    My first try with PyTorch based on "A 60 minute blitz"
    NOPE NOPE NOPE 
    not yet
'''


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    my_net = Autoencoder()
    print(my_net)

    params = list(my_net.parameters())
    print(len(params))
    print(params[0].size())
    print("---------------------------------")
