import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, transform):
        self.cifar10 = datasets.CIFAR10(root='./data',
                                        download=False,
                                        train=True,
                                        transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)


#
# def draw_random_sample(training_data, normal_training_weights, batch_size):
#     weighted_samp = WeightedRandomSampler(normal_training_weights, num_samples=len(training_data), replacement=True)
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                sampler=weighted_samp)
#     return train_loader
#
#
# dataset = MyDataset()
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# # # TODO How to get indices of images sampled based an array of weights
# w = torch.tensor([i / len(dataset) for i in range(len(dataset))])
# w = w / sum(w)
#
# train_loader = draw_random_sample(dataset, w, 2)
#
# indices_2 = []  # stores the indices of the sample based on Weighted sampler
# for batch_idx, (data, target, idx) in enumerate(train_loader):
#     indices_2.append(idx[0].item())
#     indices_2.append(idx[1].item())
#
# plt.hist(indices_2, bins=1000)
# plt.show()

w = np.array([1, 2, 3, 4])
y = np.array([True, False, True, False])
learner_err = np.dot(y, w) / np.sum(w)

print(w * y)
