import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from nets import Net3Conv
import common as comm

class WeightedLossExample(object):

    def __init__(self):
        super(WeightedLossExample, self).__init__()

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_set = torchvision.datasets.MNIST(root='dataset/', train=False, download=True, transform=transform)
        train_set = torchvision.datasets.MNIST(root='dataset/', train=True, download=True, transform=transform)

        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, num_workers=2)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net3Conv().to(self.device)

    @staticmethod
    def cross_entropy(q, p):
        # cross entropy defined by user
        q = torch.softmax(q, dim=1)
        loss_batch = -torch.sum(p * torch.log(q), dim=1)
        # print(loss_batch.shape)
        return torch.mean(loss_batch)  # using 'mean' as the reduction method

    @staticmethod
    def weighted_cross_entropy(q, p, weight):
        # weighted cross entropy defined by user
        q = torch.softmax(q, dim=1)
        loss_batch = -torch.sum(p * torch.log(q), dim=1)

        # the sum of weight should be 1
        return torch.sum(loss_batch * weight)  # using weighted average as the reduction method

    def train(self, saved_path="saved_model/net_weighted_loss.pth"):
        print("Starting training")
        learning_rate = 0.001
        num_epoch = 5

        #optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)

        self.net.train()
        criterion = nn.CrossEntropyLoss(reduction='mean')

        for epoch in range(num_epoch):
            print("epoch: %d / %d" % (epoch+1, num_epoch))
            for idx, data in enumerate(self.train_loader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = self.net(inputs)

                ce_loss_api = criterion(outputs, labels) # cross entropy provided by pytorch API

                one_hot_label = F.one_hot(labels, 10).float()
                ce_loss_defined = self.cross_entropy(outputs, one_hot_label)  # cross entropy defined by user

                weight = torch.rand([50])  # randomly generated weights, batch size is 50
                weight = weight / torch.sum(weight)
                weighted_ce_loss = self.weighted_cross_entropy(outputs, one_hot_label, weight)

                if idx == 0:
                    print("*************")
                    print("ce_loss_api:", ce_loss_api)
                    print("ce_loss_defined:", ce_loss_defined)
                    print("weighted_ce_loss", weighted_ce_loss)

                # ce_loss_defined.backward()  # using the defined loss to do back propagation
                weighted_ce_loss.backward()
                optimizer.step()

        torch.save(self.net.state_dict(), saved_path)

    def test(self, saved_path="saved_model/net_weighted_loss.pth"):
        self.net.load_state_dict(torch.load(saved_path))
        comm.accuracy(self.net, net_name="net", test_loader=self.test_loader)


if __name__ == '__main__':
    wle_obj = WeightedLossExample()
    wle_obj.train()
    # wle_obj.test()
