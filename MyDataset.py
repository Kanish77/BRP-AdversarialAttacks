from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset_name, transform_to_apply, train=True, root="./data"):
        if dataset_name == "CIFAR10":
            self.data = datasets.CIFAR10(root=root, download=True, train=train, transform=transform_to_apply)
        if dataset_name == "MNIST":
            self.data = datasets.MNIST(root=root, download=True, train=train, transform=transform_to_apply)
        if dataset_name =="FASHION-MNIST":
            self.data = datasets.FashionMNIST(root=root, download=True, train=train, transform=transform_to_apply)

    def __getitem__(self, index):
        data, target = self.data[index]
        return data, target, index

    def __len__(self):
        return len(self.data)