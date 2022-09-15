import torch
import torch.nn as nn
import torch.nn.functional as F

class BobNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        # flatten all dimensions, except the batch dimension.
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # process output of net into probabilities
        out = F.log_softmax(x, dim=1)
        return out