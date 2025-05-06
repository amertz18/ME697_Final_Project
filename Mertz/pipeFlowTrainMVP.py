import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

## CONSTANTS
PRESSURE_DROP = 0.1
RHO = 1
LENGTH = 1
DIAMETER = 0.1
EPOCHS = 3000
BATCH_SIZE = 256
MAX_NU = 0.002
LEARNING_RATE = 0.1
SOFT_WEIGHT = 2

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
        return self.fc4(X3)
    
    def pdeLoss(self, x, y, nu, rho):
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)

        U = self(torch.cat([x.unsqueeze(1), y.unsqueeze(1), nu.unsqueeze(1)], dim=1))

        ux = torch.autograd.grad(U[:, 0], x, grad_outputs=torch.ones_like(U[:, 0]), create_graph=True)[0]
        uy = torch.autograd.grad(U[:, 0], y, grad_outputs=torch.ones_like(U[:, 0]), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
        uyy = torch.autograd.grad(uy, y, grad_outputs=torch.ones_like(uy), create_graph=True)[0]

        vx = torch.autograd.grad(U[:, 1], x, grad_outputs=torch.ones_like(U[:, 1]), create_graph=True)[0]
        vy = torch.autograd.grad(U[:, 1], y, grad_outputs=torch.ones_like(U[:, 1]), create_graph=True)[0]
        vxx = torch.autograd.grad(vx, x, grad_outputs=torch.ones_like(vx), create_graph=True)[0]
        vyy = torch.autograd.grad(vy, y, grad_outputs=torch.ones_like(vy), create_graph=True)[0]

        px = torch.autograd.grad(U[:, 2], x, grad_outputs=torch.ones_like(U[:, 2]), create_graph=True)[0]
        py = torch.autograd.grad(U[:, 2], y, grad_outputs=torch.ones_like(U[:, 2]), create_graph=True)[0]

        contLoss = torch.sum((ux + vy)**2)
        momLossU = torch.sum(
            (U[:, 0] * ux + U[:, 1] * uy + px * rho**-1 - nu * (uxx + uyy))**2
        )
        momLossV = torch.sum(
            (U[:, 0] * vx + U[:, 1] * vy + py * rho**-1 - nu * (vxx + vyy))**2
        )

        return contLoss + momLossU + momLossV
    
    def bcLoss(self, X):
        """Only pass conditions on the boundary here"""

        U = self(X)

        noSlipLoss = torch.mean(U[0]**2)
        wallLoss = torch.mean(U[1]**2)

        return noSlipLoss + wallLoss
    
    def train(self, optimizer, epochs, pdeWeight):
        losses = []

        for epoch in range(epochs):

            X = torch.rand((BATCH_SIZE, 3))
            X[:, 0] = X[:, 0] * LENGTH
            X[:, 1] = X[:, 1] * DIAMETER - (DIAMETER / 2)
            X[:, 2] = X[:, 2] * MAX_NU

            XBC = X.detach().clone()
            XBC[:, 1] = torch.randint(0, 2, (BATCH_SIZE,)) * DIAMETER - (DIAMETER / 2)

            bodyLoss = self.pdeLoss(X[:, 0], X[:, 1], X[:, 2], RHO)
            wallLoss = self.bcLoss(X)

            loss = pdeWeight * bodyLoss + wallLoss
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                print(
                    f'Epoch {epoch}, PDE Loss: {bodyLoss.item()}, Boundary Loss: {wallLoss.item()}'
                )

        return losses
    
    def plotOutputSpace(self):
        x = np.linspace(0, LENGTH, 100).reshape(-1, 1)
        y = np.linspace(-DIAMETER / 2, DIAMETER / 2, 100).reshape(-1, 1)
        nu = MAX_NU / 2

        X, Y, NU = np.meshgrid(x, y, nu)
        XFlat = torch.tensor(X.flatten(), dtype = torch.float32).unsqueeze(1)
        YFlat = torch.tensor(Y.flatten(), dtype = torch.float32).unsqueeze(1)
        NUFlat = torch.tensor(NU.flatten(), dtype = torch.float32).unsqueeze(1)

        with torch.no_grad():
            UFlat = self(torch.cat([XFlat, YFlat, NUFlat], dim=1)).numpy()
        U = UFlat.reshape((3, 100, 100))

        plt.figure(figsize=(10, 6))
        plt.contourf(X[:, :, 0], Y[:, :, 0], U[0], levels = 100, cmap=plt.cm.jet)
        plt.colorbar(label='Streamwise Velocity (u)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Streamwise Velocity')
        plt.savefig('fig/CircularPipeFlowU.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.contourf(X[:, :, 0], Y[:, :, 0], U[1], levels = 100, cmap=plt.cm.jet)
        plt.colorbar(label='Spanwise Velocity (v)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Spanwise Velocity')
        plt.savefig('fig/CircularPipeFlowV.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.contourf(X[:, :, 0], Y[:, :, 0], U[1], levels = 100, cmap=plt.cm.jet)
        plt.colorbar(label='Centerline Pressure Profile (p)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Centerline Pressure Profile')
        plt.savefig('fig/CircularPipeFlowP.png')
        plt.close()

pipeFlowModel = circlePipePINN()
optimizer = optim.Adam(pipeFlowModel.parameters(), lr = LEARNING_RATE)

trainingLoss = pipeFlowModel.train(optimizer, EPOCHS, SOFT_WEIGHT)

pipeFlowModel.plotOutputSpace()

plt.figure(figsize=(10, 6))
plt.semilogy(range(EPOCHS), trainingLoss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Total Loss vs Epoch (Adam Optimizer)')
plt.grid()
plt.savefig('fig/CircularPipeFlowLoss.png')
plt.close()