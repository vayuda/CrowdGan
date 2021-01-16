import torch
import torch.nn as nn


class CrowdLayer(nn.Module):
    def __init__(self, output_dim, num_annotators):
        super().__init__()
        self.num_annotators = num_annotators
        self.output_dim = output_dim
        weights = torch.zeros((output_dim, output_dim, num_annotators))
        self.weights = nn.Parameter(weights, requires_grad=True)

        # initialize weights as identity matrix
        for row in range(len(self.weights)):
            nn.init.ones_(self.weights[row][:][row])

    def forward(self, x):
        return torch.einsum('ij,kjl->ikl', x, self.weights)
