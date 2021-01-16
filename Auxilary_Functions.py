import torch
import torch.nn as nn


def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


def sma(arr, n):
    return [sum(arr[i - n:i]) / n for i in range(n, len(arr))]


def masked_cross_entropy(output, target, ce_loss):
    # if annotation does not exist then don't count it
    mask = target != -1
    output = torch.transpose(output, 1, 2)
    loss = ce_loss(output[mask], target[mask])
    return loss