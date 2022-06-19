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
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net3Conv().to(self.device)

    # ---------------------------------------------------------------------------------------------#
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

    # ---------------------------------------------------------------------------------------------#
    @staticmethod
    def kl_div(input, target):
        input = F.log_softmax(input, dim=1)
        loss_pointwise = target * (target - input)
        return torch.mean(loss_pointwise)

    @staticmethod
    def kl_div_weighted(input, target, weight):
        input = F.log_softmax(input, dim=1)
        loss_pointwise = target * (target - input)
        values, _ = torch.max(loss_pointwise, 1)
        return torch.sum(values * weight)

    # ---------------------------------------------------------------------------------------------#
    @staticmethod
    def mulit_margin_loss(input, target):
        loss_arr = []
        for i, output in enumerate(input):
            cur_target = target[i]
            loss = sum(max(0, 1 - output[cur_target] + output[k]) for k in range(len(output)))
            loss = loss - max(0, 1 - output[cur_target] + output[cur_target])
            loss = loss / len(output)
            loss_arr.append(loss)
        loss_arr = torch.tensor(loss_arr, requires_grad=True)
        return torch.mean(loss_arr)

    @staticmethod
    def mulit_margin_loss_weighted(input, target, weight):
        loss_arr = []
        for i, output in enumerate(input):
            cur_target = target[i]
            loss = sum(max(0, 1 - output[cur_target] + output[k]) for k in range(len(output)))
            loss = loss - max(0, 1 - output[cur_target] + output[cur_target])
            loss = loss / len(output)
            loss_arr.append(loss)
        loss_arr = torch.tensor(loss_arr, requires_grad=True).to("cuda")
        return torch.sum(loss_arr * weight)

    # ---------------------------------------------------------------------------------------------#

    def train(self, loss_name, saved_path="saved_model/net_weighted_loss.pth"):
        print("Starting training")
        learning_rate = 0.001
        num_epoch = 5

        #optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self.net.train()
        # if loss_name == "CE":
        #     criterion = nn.CrossEntropyLoss(reduction='mean')
        # elif loss_name == "KL":
        #     criterion = nn.KLDivLoss(reduction="mean")
        # elif loss_name == "MM":
        #     criterion = nn.MultiMarginLoss(reduction="mean")
        criterion_ce = nn.CrossEntropyLoss(reduction="mean")
        criterion_kl = nn.KLDivLoss(reduction="batchmean")


        for epoch in range(num_epoch):
            print("epoch: %d / %d" % (epoch+1, num_epoch))
            for idx, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.net(inputs)
                weight = torch.rand([100]).to(self.device)  # randomly generated weights, batch size is 50
                weight = weight / torch.sum(weight)

                with torch.no_grad():
                    self.net.eval()
                    loss_ce = criterion_ce(outputs, labels)
                    one_hot_label = F.one_hot(labels, 10).float()
                    loss_kl = criterion_kl(F.log_softmax(outputs), one_hot_label)
                self.net.train()
                loss = criterion_ce(outputs, labels)
                # if loss_name == "CE":
                #     loss_api = criterion(outputs, labels) # cross entropy provided by pytorch API
                #     one_hot_label = F.one_hot(labels, 10).float()
                #     loss_self_impl = self.cross_entropy(outputs, one_hot_label)  # cross entropy defined by user
                #     weighted_loss = self.weighted_cross_entropy(outputs, one_hot_label, weight)
                # elif loss_name == "KL":
                #     one_hot_label = F.one_hot(labels, 10).float()
                #     loss_api = criterion(F.log_softmax(outputs), one_hot_label)
                #     loss_self_impl = self.kl_div(outputs, one_hot_label)
                #     weighted_loss = self.kl_div_weighted(outputs, one_hot_label, weight)
                # elif loss_name == "MM":
                #     loss_api = criterion(outputs, labels)
                #     #loss_self_impl = self.mulit_margin_loss(outputs, labels)
                #     #weighted_loss = self.mulit_margin_loss_weighted(outputs, labels, weight)

                if idx % 150 == 0:
                    print("*************")
                    print("loss_ce:", loss_ce)
                    print("loss_kl:", loss_kl)
                    # print("loss_defined:", loss_self_impl)
                    # print("weighted_loss", weighted_loss)

                # ce_loss_defined.backward()  # using the defined loss to do back propagation
                loss.backward()
                optimizer.step()

        torch.save(self.net.state_dict(), saved_path)

    def test(self, saved_path="saved_model/net_weighted_loss.pth"):
        self.net.load_state_dict(torch.load(saved_path))
        comm.accuracy(self.net, net_name="net", test_loader=self.test_loader)

def return_accuracy(model, device, loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')


if __name__ == '__main__':
    wle_obj = WeightedLossExample()
    wle_obj.train(loss_name="MM")
    return_accuracy(wle_obj.net, wle_obj.device, wle_obj.test_loader)