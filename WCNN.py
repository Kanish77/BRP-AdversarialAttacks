import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from MyDataset import CustomDataset
from advertorch.context import ctx_noparamgrad_and_eval


# Based on https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
# and on nets.py file. Is a weak CNN learner designed primarily for MNIST. But can be extended for CIFAR.
from adversary_attack import AdversaryCode


class WCNN3(nn.Module):  # 3 layers has structure: 2 conv layers, and 1 fully connected
    def __init__(self):
        super(WCNN3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.out = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # flatten the output of conv2 to (batch_size, 32 * 4 * 4)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class CNNBaseLearner(object):
    @staticmethod
    def weighted_cross_entropy(q, p, weight):
        # weighted cross entropy defined by user
        q = torch.softmax(q, dim=1)
        loss_batch = -torch.sum(p * torch.log(q), dim=1)

        # the sum of weight should be 1 thus we divide by sum(weight)
        return torch.sum(loss_batch * weight) / torch.sum(weight)  # using weighted average as the reduction method

    @staticmethod
    def perform_training(model, device, train_loader, num_epochs, learning_rate, weights, adversarial_training,
                         adversary_algo=None):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        n_total_steps = len(train_loader)
        print_once = True
        for epoch in range(num_epochs):
            for i, (images, labels, index) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                index = index.to(device)

                # If we are performing adversarial_training, then we need to perturb the input image
                if adversarial_training:
                    with ctx_noparamgrad_and_eval(model):
                        images = adversary_algo.perturb(images, labels)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)

                # Computing weighted loss
                one_hot_label = F.one_hot(labels, 10).float()
                corresponding_weights = torch.index_select(weights, 0, index)
                weighted_ce_loss = CNNBaseLearner.weighted_cross_entropy(outputs, one_hot_label, corresponding_weights)

                # Backward and optimize
                weighted_ce_loss.backward()
                optimizer.step()

                if (i + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], '
                          f'C-loss: {weighted_ce_loss.item():.4f}')
                if print_once:
                    imgs = images.to("cpu")
                    for i in range(3):
                        img = imgs[i].reshape(28, 28)
                        plt.subplot(2, 3, i + 1)
                        plt.imshow(img, cmap='gray')
                        plt.title("p-train")
                    print_once = False
                    plt.show()
        return model

    @staticmethod
    def compute_incorrect_array_adversary_training(model, device, train_loader, adversary_algo):
        model.eval()
        n_correct = 0
        n_samples = 0
        do_once = True
        for (images, labels, idx) in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = adversary_algo.perturb(images, labels)
            with torch.no_grad():
                outputs = model(images)  # result of doing forward pass on these inputs
            # max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            if do_once:
                incorrect_array = predicted != labels
                do_once = False
            else:
                incorrect_array = torch.cat((incorrect_array, predicted != labels))
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
        return incorrect_array

    @staticmethod
    def compute_incorrect_array(model, device, train_loader):
        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            do_once = True
            for (images, labels, idx) in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)  # result of doing forward pass on these inputs
                # max returns (value, index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                if do_once:
                    incorrect_array = predicted != labels
                    do_once = False
                else:
                    incorrect_array = torch.cat((incorrect_array, predicted != labels))
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')
            return incorrect_array

    @staticmethod
    def predict_test_set(model, device, test_dataset):
        model.eval()
        with torch.no_grad():
            predictions = []
            # correct = 0
            for image, label, idx in test_dataset:
                image = image.to(device)
                output = model(image)
                pred = output.argmax(dim=1, keepdim=False).item()
                predictions.append(pred)
            # print(100. * correct / len(test_dataset))
            return np.array(predictions)

    @staticmethod
    def return_accuracy(model, device, loader):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels, idx in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')

    @staticmethod
    def return_adversarial_robustness(model, device, loader, adversary_algo):
        model.eval()
        n_correct = 0
        do_once = True
        for (images, labels, index) in loader:
            images, labels, index = images.to(device), labels.to(device), index.to(device)

            images = adversary_algo.perturb(images, labels)
            with torch.no_grad():
                outputs = model(images)  # result of doing forward pass on these inputs
            # max returns (value, index)
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            n_correct += pred.eq(labels.view_as(pred)).sum().item()

            if do_once:
                images = images.to("cpu")
                for i in range(3):
                    img = images[i].reshape(28, 28)
                    plt.subplot(2, 3, i + 1)
                    plt.imshow(img, cmap='gray')
                    plt.title("perturbed images")
                do_once = False
                plt.show()

        acc = 100. * n_correct / len(loader.dataset)
        print(f'Adversarial attack accuracy of the network: {acc} %')


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mnist_train = CustomDataset(dataset_name="MNIST", transform_to_apply=transforms.ToTensor(), train=True)
# mnist_test = CustomDataset(dataset_name="MNIST", transform_to_apply=transforms.ToTensor(), train=False)
# train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1000, shuffle=True)
#
# wcnn = WCNN3().to(device)
# N = len(mnist_train)
# w = torch.tensor([1/N for i in range(N)])
#
# adversary_algo = AdversaryCode("", wcnn).get_adversary_method("FGSM")
#
# wcnn = CNNBaseLearner.perform_training(wcnn, device, train_loader, 1, 0.001, w, True, adversary_algo)
# CNNBaseLearner.return_accuracy(wcnn, device, test_loader)
# CNNBaseLearner.return_adversarial_robustness(wcnn, device, test_loader, adversary_algo)