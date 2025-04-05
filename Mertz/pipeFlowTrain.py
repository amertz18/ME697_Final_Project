import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

## CONSTANTS
PRESSURE_DROP = 0.1
RHO = 1
LENGTH = 1
DIAMETER = 0.1

class inputNN(nn.Module):
    def __init__(self, inputLen):
        super(inputNN, self).__init__()
        self.fc1 = nn.Linear(inputLen, 50)
        self.fc1 = nn.Linear(50, 50)
        self.fc1 = nn.Linear(50, 50)
        self.fc1 = nn.Linear(50, 3)

    def forward(self, X):
        X1 = F.sigmoid(self.fc1(X))
        X2 = F.sigmoid(self.fc2(X1))
        X3 = F.sigmoid(self.fc3(X2))
        X4 = F.sigmoid(self.fc4(X3))
        return self.fc4(X4)
    
    def loss(self, X):
        outputs = self(X)

        uLoss = (outputs[0] - 

class circlePipePINN(nn.Module):
    def __init__(self):
        super(circlePipePINN, self).__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 3)

    def forward(self, X):
        X1 = F.sigmoid(self.fc1(X))
        X2 = F.sigmoid(self.fc2(X1))
        X3 = F.sigmoid(self.fc3(X2))
        X4 = F.sigmoid(self.fc4(X3))
        return self.fc4(X4)