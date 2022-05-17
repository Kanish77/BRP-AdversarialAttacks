import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from MyDataset import CustomDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

    def perform_training(self, train_loader, num_epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels, index) in enumerate(train_loader):
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)

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

    def compute_incorrect_array(self, train_loader):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            do_once = True
            for (images, labels, idx) in train_loader:
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)
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

    """
    Return 1d numpy array of predictions. Where each entry in the array corresponds to the prediction for test image
    """
    def predict_test_set(self, test_dataset):
        self.eval()
        with torch.no_grad():
            predictions = []
            #correct = 0
            for image, label, idx in test_dataset:
                image = image.reshape(-1, 28 * 28).to(device)
                output = self(image)
                pred = output.argmax(dim=1, keepdim=False).item()
                predictions.append(pred)
            #print(100. * correct / len(test_dataset))
            return np.array(predictions)

    def return_accuracy(self, test_loader):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels, idx in test_loader:
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)
                outputs = self(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')


# Hyper-parameters
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = CustomDataset("MNIST", transforms.ToTensor(), True)
test_dataset = CustomDataset("MNIST", transforms.ToTensor(), False)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
mlp = MLP(784, 10, 10).to(device)
mlp.perform_training(train_loader, 1, 0.001)
mlp.return_accuracy(test_loader)
res = mlp.predict_test_set(test_dataset)
print(res)