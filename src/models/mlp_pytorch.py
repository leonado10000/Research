import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.net(X)


# ----- Data -----
data = MNIST("./src/data", train=True, transform=ToTensor(), download=False)
loader = DataLoader(data, batch_size=32, shuffle=True)

# ----- Model -----
mlp = MLP([784, 128, 10])
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
n = 1
# ----- Training -----
for epoch in range(1):
    for X, y in loader:
        X = X.view(X.size(0), -1)

        optimizer.zero_grad()

        output = mlp(X)
        loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()

        if n%100 == 0:
            print(f"Epoch {n}, Loss: {loss.item():.4f}")

        n += 1