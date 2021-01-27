import torch
import torch.nn as nn
import numpy as np
from CrowdLayer import CrowdLayer

class BaseModel(nn.Module):
    def __init__(self, input_size, output_size, ngpu=1):
        super().__init__()
        self.input_size = input_size
        print(input_size)
        self.output_size = output_size
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(np.prod(input_size), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)


class CrowdNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_annotators, ngpu=1, device = "cuda:0"):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ngpu = ngpu
        self.device = device
        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = nn.Linear(np.prod(input_size), 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(.5)
        self.linear2 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.crowd_layer = CrowdLayer(output_size, num_annotators)

    def forward(self, x, predict=False):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.softmax(x)
        if predict:
            return x
        return self.crowd_layer(x)


class DiscriminatorSLWAE2(nn.Module):
    """
    Has Seperate Linear Weights for each class and Annotator Embeddings with higher dim than v1
    """

    def __init__(self, n_instances, n_hidden, n_annotators, n_classes, ngpu=1, device="cuda:0"):
        super().__init__()
        self.n_annotators = n_annotators
        self.ngpu = ngpu
        self.device = device
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_instances = n_instances
        self.aedims = 128
        self.lin2h = 32

        self.lin1 = nn.Linear(n_instances * n_classes, n_hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin2 = nn.Linear(64 + self.aedims, self.lin2h)
        self.sigmoid = nn.Sigmoid()
        self.lin3 = nn.Linear(self.lin2h, 1)
        self.linae = nn.Linear(n_annotators, self.aedims)
        self.norm1 = nn.BatchNorm2d(self.aedims)
        self.dis = nn.Sequential(
            nn.Linear(n_instances * n_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.dis(x)
        x = torch.cat((x, self.linae(torch.eye(self.n_annotators).to(self.device))), 1)
        x = self.lin2(self.norm1(x))
        x = self.relu(x)

        x = self.sigmoid(self.lin3(x))
        return x

    def __str__(self):
        return "DiscriminatorSLWAE"

class DiscriminatorSLWAE(nn.Module):
    """
    Has Seperate Linear Weights for each class and Annotator Embeddings
    """

    def __init__(self, n_instances, n_hidden, n_annotators, n_classes, ngpu=1, device="cuda:0"):
        super().__init__()
        self.n_annotators = n_annotators
        self.ngpu = ngpu
        self.device = device
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_instances = n_instances

        self.lin1 = nn.Linear(n_instances * n_classes, n_hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin2 = nn.Linear(64 + n_annotators, 32)
        self.sigmoid = nn.Sigmoid()
        self.lin3 = nn.Linear(32, 1)

        self.dis = nn.Sequential(
            nn.Linear(n_instances * n_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.dis(x)
        x = torch.cat((x, torch.eye(self.n_annotators).to(self.device)), 1)
        x = self.lin2(x)
        x = self.relu(x)

        x = self.sigmoid(self.lin3(x))
        return x

    def __str__(self):
        return "DiscriminatorSLWAE"


class DiscriminatorCE(nn.Module):
    def __init__(self, n_instances, n_hidden, n_annotators, n_classes, ngpu=1,device = "cuda:0"):
        '''
        input size is (59,8,n_instances)
        concat class embeddings so new shape is (59,8,n_instances + n_classes)
        pass through a linear layer
        Afterwards it will be 59, 8, n_hidden
        Add the class representation vectors together
        Aftewards it will be 59, hidden
        Pass through second linear layer
        sigmoid activation
        '''
        super().__init__()
        self.n_annotators = n_annotators
        self.ngpu = ngpu
        self.device =device
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_instances = n_instances

        self.lin1 = nn.Linear(n_instances + n_classes, n_hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.cat((x,torch.eye(self.n_classes).repeat(self.n_annotators,1,1).to(self.device)), 2)
        x = self.relu(self.lin1(x))
        x = self.drop(x)
        x = torch.sum(x,1)
        x = self.sigmoid(self.lin2(x))
        return x


class DiscriminatorSLW(nn.Module):
    def __init__(self, n_instances, n_hidden, n_annotators, n_classes, ngpu=1, device="cuda:0"):
        '''
        input size is (59,8,64)
        have a linear layer for each class
        afterwards it will be (59, 8, n_hidden)
        add the class representation vectors together
        aftewards it will be (59, n_hidden)
        linear layer output to (59,1) and sigmoid
        '''
        super().__init__()
        self.n_annotators = n_annotators
        self.ngpu = ngpu
        self.device = device
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_instances = n_instances

        self.lin1 = nn.Parameter(torch.zeros(n_classes, n_instances, n_hidden), requires_grad= True)
        nn.init.xavier_uniform_(self.lin1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.einsum("ijk,jkl->ijl",x,self.lin1)
        x = self.drop(self.relu(x))
        x = torch.mean(x, 1)
        x = self.sigmoid(self.lin2(x))
        return x
    def __str__(self):
        return "DiscriminatorSLW"


class DiscriminatorSLWCE(nn.Module):
    def __init__(self, n_instances, n_hidden, n_annotators, n_classes, ngpu=1, device="cuda:0"):
        '''
        has both seperate linear layers for each class and uses class embeddings
        '''
        super().__init__()
        self.n_annotators = n_annotators
        self.ngpu = ngpu
        self.device =device
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_instances = n_instances

        self.lin1 = nn.Parameter(torch.zeros(n_classes, n_instances + n_classes, n_hidden), requires_grad= True)
        nn.init.xavier_uniform_(self.lin1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.cat((x, torch.eye(self.n_classes).repeat(self.n_annotators, 1, 1).to(self.device)), 2)
        x = torch.einsum("ijk,jkl->ijl",x,self.lin1)
        x = self.drop(self.relu(x))
        x = torch.mean(x, 1)
        x = self.sigmoid(self.lin2(x))
        return x

    def __str__(self):
        return "DiscriminatorSLWCE"


class DiscriminatorE(nn.Module):
    def __init__(self, n_instances, n_hidden, n_annotators, n_classes, ngpu=1, device="cuda:0"):
        '''
        has seperate linear layers for each class uses both class and user embeddings
        '''
        super().__init__()
        self.n_annotators = n_annotators
        self.ngpu = ngpu
        self.device =device
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_instances = n_instances

        self.lin1 = nn.Parameter(torch.zeros(n_classes, n_instances + n_classes, n_hidden), requires_grad=True)
        nn.init.xavier_uniform_(self.lin1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin2 = nn.Linear(n_hidden + n_annotators, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.cat((x, torch.eye(self.n_classes).repeat(self.n_annotators, 1, 1).to(self.device)), 2)
        x = torch.einsum("ijk,jkl->ijl",x,self.lin1)
        x = self.drop(self.relu(x))
        x = torch.mean(x, 1)
        x = torch.cat((x,torch.eye(self.n_annotators).to(self.device)), 1)
        x = self.sigmoid(self.lin2(x))
        return x
    def __str__(self):
        return "DiscriminatorE"



