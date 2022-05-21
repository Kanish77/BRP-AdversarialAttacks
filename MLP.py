import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from MyDataset import CustomDataset
from base_learner import BaseLearner
from advertorch.context import ctx_noparamgrad_and_eval


class MLP(nn.Module, BaseLearner):
    def __init__(self, input_size, hidden_size, num_classes, device):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

    @staticmethod
    def weighted_cross_entropy(q, p, weight):
        # weighted cross entropy defined by user
        q = torch.softmax(q, dim=1)
        loss_batch = -torch.sum(p * torch.log(q), dim=1)

        # the sum of weight should be 1 thus we divide by sum(weight)
        return torch.sum(loss_batch * weight) / torch.sum(weight)  # using weighted average as the reduction method

    def perform_training(self, train_loader, num_epochs, learning_rate, weights, adversarial_training,
                         adversary_algo=None):
        self.train()
        # criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels, index) in enumerate(train_loader):
                # origin shape: [100, 1, 28, 28], resized: [100, 784]
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)

                # If we are performing adversarial_training, then we need to perturb the input image
                if adversarial_training:
                    with ctx_noparamgrad_and_eval(self):
                        images = adversary_algo.perturb(images, labels)

                # Forward pass
                optimizer.zero_grad()
                outputs = self(images)

                # Computing weighted loss
                one_hot_label = F.one_hot(labels, 10).float()
                corresponding_weights = torch.index_select(weights, 0, index)
                weighted_ce_loss = self.weighted_cross_entropy(outputs, one_hot_label, corresponding_weights)

                # Backward and optimize
                weighted_ce_loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], '
                          f'C-loss: {weighted_ce_loss.item():.4f}')
        # Save the model
        # PATH = './cnn.pth'
        # torch.save(self.state_dict(), PATH)
        return self


    def compute_incorrect_array_adversary_training(self, train_loader, adversary_algo):
        self.eval()
        n_correct = 0
        n_samples = 0
        do_once = True
        for (images, labels, idx) in train_loader:
            images = images.reshape(-1, 28 * 28).to(self.device)
            labels = labels.to(self.device)

            images = adversary_algo.perturb(images, labels)
            with torch.no_grad():
                outputs = self(images)  # result of doing forward pass on these inputs
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



    def compute_incorrect_array(self, train_loader):
        self.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            do_once = True
            for (images, labels, idx) in train_loader:
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)  # result of doing forward pass on these inputs
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

    """
    Return 1d numpy array of predictions. Where each entry in the array corresponds to the prediction for test image
    """

    def predict_test_set(self, test_dataset):
        self.eval()
        with torch.no_grad():
            predictions = []
            # correct = 0
            for image, label, idx in test_dataset:
                image = image.reshape(-1, 28 * 28).to(self.device)
                output = self(image)
                pred = output.argmax(dim=1, keepdim=False).item()
                predictions.append(pred)
            # print(100. * correct / len(test_dataset))
            return np.array(predictions)

    def return_accuracy(self, test_loader):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels, idx in test_loader:
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')

# # Hyper-parameters
# num_epochs = 2
# batch_size = 100
# learning_rate = 0.001
#
# # MNIST dataset
# train_dataset = CustomDataset("MNIST", transforms.ToTensor(), True)
# test_dataset = CustomDataset("MNIST", transforms.ToTensor(), False)
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mlp = MLP(784, 10, 10, device).to(device)
# mlp.perform_training(train_loader, 1, 0.001)
# mlp.return_accuracy(test_loader)
# res = mlp.predict_test_set(test_dataset)
# print(res)
