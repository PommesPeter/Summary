import torch
import torchvision
import numpy as np
from utils import *
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.nn import functional as f

learning_rate = 0.01
batch_size = 512

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_data = torchvision.datasets.MNIST('./data', train=True, transform=transforms, download=True)
test_data = torchvision.datasets.MNIST('./data', train=False, transform=transforms, download=True)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)