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


class CustomDataset(Dataset):
    def __init__(self, dataset_name, transform_to_apply, train=True, root="./data"):
        if dataset_name == "CIFAR10":
            self.data = datasets.CIFAR10(root=root, download=True, train=train, transform=transform_to_apply)
        if dataset_name == "MNIST":
            self.data = datasets.MNIST(root=root, download=True, train=train, transform=transform_to_apply)
        if dataset_name == "FASHION-MNIST":
            self.data = datasets.FashionMNIST(root=root, download=True, train=train, transform=transform_to_apply)

    def __getitem__(self, index):
        data, target = self.data[index]
        return data, target, index

    def __len__(self):
        return len(self.data)


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


data = CustomDataset(dataset_name="CIFAR10", transform_to_apply=transforms.ToTensor(), train=False)
print_once = True

temp = np.array([[0], [1]]) == np.array([0, 1, 0])
print(temp)
print(temp.T)

temp2 = np.array([0, 1, 0]) == np.array([[0], [1]])
print(temp2)
print(temp2.T * 0.4)

one = np.array([0, 1, 5])
two = np.array([0, 5, 5])
print(one != two)
