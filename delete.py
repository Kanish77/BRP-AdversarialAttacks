from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from matplotlib import pyplot as plt


class PytorchNet(nn.Module):
    def __init__(self):
        super(PytorchNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default="adv", help="cln | adv")
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.mode == "cln":
        flag_advtrain = False
        nb_epoch = 10
        model_filename = "mnist_lenet5_clntrained.pt"
    elif args.mode == "adv":
        flag_advtrain = True
        nb_epoch = 90
        model_filename = "mnist_lenet5_advtrained.pt"
    else:
        print("IDFK")

    train_loader = get_mnist_train_loader(
        batch_size=args.train_batch_size, shuffle=True)
    test_loader = get_mnist_test_loader(
        batch_size=args.test_batch_size, shuffle=False)

    #model = LeNet5()
    model = PytorchNet().to(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if flag_advtrain:
        from advertorch.attacks import LinfPGDAttack, GradientSignAttack

        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=False)
        adversary_train = GradientSignAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.5, targeted=False)
    print_once = True
    for epoch in range(nb_epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ori = data
            if flag_advtrain:
                # when performing attack, the model needs to be in eval mode
                # also the parameters should be accumulating gradients
                with ctx_noparamgrad_and_eval(model):
                    data = adversary_train.perturb(data, target)

            optimizer.zero_grad()
            output = model(data)
            # loss = F.cross_entropy(
            #     output, target, reduction='elementwise_mean')
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

            if print_once:
                imgs = data.to("cpu")
                for i in range(3):
                    img = imgs[i].reshape(28, 28)
                    plt.subplot(2, 3, i + 1)
                    plt.imshow(img, cmap='gray')
                    plt.title("perturbed images")
                print_once = False
                plt.show()

        model.eval()
        test_clnloss = 0
        clncorrect = 0
        do_once = True

        if flag_advtrain:
            test_advloss = 0
            advcorrect = 0

        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)
            with torch.no_grad():
                output = model(clndata)
            test_clnloss += F.nll_loss(output, target, reduction="sum").item()
            #test_clnloss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

            if flag_advtrain:
                advdata = adversary_train.perturb(clndata, target)
                with torch.no_grad():
                    output = model(advdata)
                test_advloss += F.nll_loss(output, target, reduction='sum').item()
                #test_advloss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                advcorrect += pred.eq(target.view_as(pred)).sum().item()
                if do_once:
                    advdata = advdata.to("cpu")
                    for i in range(3):
                        img = advdata[i].reshape(28, 28)
                        plt.subplot(2, 3, i + 1)
                        plt.imshow(img, cmap='gray')
                        plt.title("perturbed images")
                    do_once = False
                    plt.show()

        test_clnloss /= len(test_loader.dataset)
        print('\nTest set: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss, clncorrect, len(test_loader.dataset),
                  100. * clncorrect / len(test_loader.dataset)))
        if flag_advtrain:
            test_advloss /= len(test_loader.dataset)
            print('Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
                      test_advloss, advcorrect, len(test_loader.dataset),
                      100. * advcorrect / len(test_loader.dataset)))

    torch.save(
        model.state_dict(),
        os.path.join(TRAINED_MODEL_PATH, model_filename))