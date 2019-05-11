#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch

'''
    https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
    a 60 minute blitz
'''

'''  what is pytorch?  '''
x = torch.empty(5, 3)
print(x)

print()
x = torch.rand(5, 3)
print(x)

print("---------------------------------")
# bridge
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
np.add(a, 1, out=a)
print(a)
print(b)

print("---------------------------------")
# cuda tensor
# let us run this cell only if cuda is available
# we will use ``torch.device`` objects to move tensors in and out of gpu
if torch.cuda.is_available():
    device = torch.device("cuda")          # a cuda device object
    print(device)
    y = torch.ones_like(x, device=device)  # directly create a tensor on gpu
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

print()
print("---------------------------------")
print("---------------------------------")


'''  autograd: automatic differentiation  '''
x = torch.ones(2, 2, requires_grad = True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)
print(out)

print("---------------------------------")
# changing tensors request_grad flag
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

print("---------------------------------")
# gradients
x = torch.ones(2, 2, requires_grad = True)
y = x + 2
z = y * y * 3
out = z.mean()
print("out")
print(out)
out.backward()
print(x.grad) # d(out)/dx

print("---------------------------------")
x = torch.randn(3, requires_grad=True)

y = x * 2

while y.data.norm() < 1000:
    # .data.norm() - normalizacja danych sqrt(x1^2 + x2^2 + ...)
    y = y * 2

print(y)
# y is vector

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print("---------------------------------")
# Stoping autograd from tracking information of tensor
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)