import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data import WeightedRandomSampler
from MyDataset import CustomDataset

# Structure based on https://github.com/python-engineer/pytorchTutorial/blob/master/14_cnn.py
# Is made for the CIFAR dataset

class ConvNet(nn.Module):
    def __init__(self, device):
        super(ConvNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 20, 5, 1)
        self.conv1 = nn.Conv2d(20, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.device = device

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # def __init__(self, device):
    #     super(ConvNet, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)
    #     self.device = device
    #
    # def forward(self, x):
    #     # -> n, 3, 32, 32
    #     x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
    #     x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
    #     x = x.view(-1, 16 * 5 * 5)  # -> n, 400
    #     x = F.relu(self.fc1(x))  # -> n, 120
    #     x = F.relu(self.fc2(x))  # -> n, 84
    #     x = self.fc3(x)  # -> n, 10
    #     return x

    def perform_training(self, train_loader, num_epochs, learning_rate):
        self.train()
        criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        n_total_steps = len(train_loader)
        print("hi there")
        print(len(train_loader))
        for epoch in range(num_epochs):
            for i, (images, labels, index) in enumerate(train_loader):
                # origin shape: [4, 3, 32, 32] = 4, 3, 1024
                # input_layer: 3 input channels, 6 output channels, 5 kernel size
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
        # Save the model
        # PATH = './cnn.pth'
        # torch.save(self.state_dict(), PATH)
        return self

    """
    
    """

    def compute_incorrect_array(self, train_loader):
        self.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            do_once = True
            for (images, labels, idx) in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)  # result of doing forward pass on these inputs
                # max returns (value, index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                if do_once:
                    incorrect_array = predicted != labels
                    print(incorrect_array)
                    do_once = False
                else:
                    incorrect_array = torch.cat((incorrect_array, predicted != labels))
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')
            return incorrect_array
        # with torch.no_grad():
        #     bool_arr = []
        #     for (images, labels, idx) in train_loader:
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         outputs = self(images) # result of doing forward pass on these inputs
        #         # max returns (value, index)
        #         _, predicted = torch.max(outputs, 1)
        #         incorrect = (predicted != labels)
        #         for i in incorrect:
        #             bool_arr.append(i.item())
        #     return torch.tensor(bool_arr)

    """
    Return 1d numpy array of predictions. Where each entry in the array corresponds to the prediction for test image
    """

    def predict_test_set(self, test_dataset):
        self.eval()
        with torch.no_grad():
            predictions = []
            for image, label, idx in test_dataset:
                image = image.to(self.device)
                output = self(image)
                pred = output.argmax(dim=1, keepdim=True).item()
                predictions.append(pred)
            return predictions

    def return_accuracy(self, data_loader):
        print_once = True

        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            for (images, labels, idx) in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)  # result of doing forward pass on these inputs
                # max returns (value , index), where value is the maximum value in the array along the given dimension
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                if print_once:
                    print("Correct labels", labels)
                    print("Outputs", outputs)
                    value, predicted = torch.max(outputs, 1)
                    print("value", value)
                    print("predicted", predicted)
                    print("n_samples", n_samples)
                    print("n_correct", n_correct)
                    print_once = False
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')

    def evaluate_performance_per_class(self, test_loader, batch_size, classes):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for (images, labels, idx) in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')

            for i in range(10):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {classes[i]}: {acc} %')

# # Hyper-parameters
# batch_size = 4
#
# # dataset has PILImage images of range [0, 1].
# # We transform them to Tensors of normalized range [-1, 1]
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# # CIFAR10: 60,000 32x32 color images in 10 classes, with 6000 images per class
# train_dataset = CustomCifar(transform_to_apply=transform, train=True)
# test_dataset = CustomCifar(transform_to_apply=transform, train=False)
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# # image, label = train_dataset[1]
# # print(label)
# # imshow(torchvision.utils.make_grid(image))
# # print(image)
#
# #
# # # get some random training images
# # dataiter = iter(train_loader)
# # images, labels = dataiter.next()
# #
# # # show images
# # imshow()
#
#
# ## RANDOM tests with duplication and training
# model_good = ConvNet().to(device)
# model_bad = copy.deepcopy(model_good)
#
# arr = []
# arr.append(model_good)
# arr.append(model_bad)
#
#
# # train both models
# #arr[0].perform_training(train_loader, 1, 0.001)
# arr[1].perform_training(train_loader, 0, 0.001)
#
# print("performance of good model")
# res = arr[0].compute_incorrect_array(train_loader)
# print("\n", "\n")
# print(res)
# # print("performance of bad model")
# # arr[1].evaluate_performance(test_loader)
#
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# train_dataset = CustomDataset("CIFAR10", transform, True)
# test_dataset = CustomDataset("CIFAR10", transform, False)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=4,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=4,
#                                           shuffle=False)
# cnn = ConvNet().to(device)
# cnn.perform_training(train_loader, 2, 0.001)
# cnn.return_accuracy(test_loader)
