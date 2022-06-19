import torch
import torch.nn.functional as F

@staticmethod
def cross_entropy(q, p):
    # cross entropy defined by user
    q = torch.softmax(q, dim=1)
    loss_batch = -torch.sum(p * torch.log(q), dim=1)
    # print(loss_batch.shape)
    return torch.mean(loss_batch)  # using 'mean' as the reduction method

@staticmethod
def weighted_cross_entropy(input, target, weight):
    # weighted cross entropy defined by user
    input = torch.softmax(input, dim=1)
    loss_batch = -torch.sum(target * torch.log(input), dim=1)

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
    loss_arr = torch.tensor(loss_arr, requires_grad=True)
    return torch.sum(loss_arr * weight)