import sys
sys.path.append(".")
from src.core.tensor import Tensor
from src.core.meanSquareError import MSE
from src.core.linear import Linear
from src.core.oneHot import oneHotEncoding
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class MLP:
    def __init__(self, layer_sizes):
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
            # if i < len(layer_sizes) - 2:
            #     layers.append(nn.ReLU())
        self.net = [layer for layer in layers]
    
    def forward(self, X):
        for i, layer in enumerate(self.net):
            X = layer.forward(X)
            if i<len(self.net):
                X = X.relu()
        return X


# ----- Data -----
data = MNIST("./src/data", train=True, transform=ToTensor(), download=False)
# loader = DataLoader(data, batch_size=32, shuffle=True)

# ----- Model -----
mlp = MLP([784, 128, 10])
loss_fn = MSE()

# ----- Training -----
for epoch in range(1):
    n = 1
    for X, y in data:
        X = Tensor(X.tolist())
        X = X.view([1,784])
        y = oneHotEncoding(y, 10)
        y = y.view([1,10])
        output = mlp.forward(X)
        loss = loss_fn(output, y)
        loss.backward()

        if n%10 == 0:
            print(f"Epoch {epoch}: {n}, Loss: {loss.item()}")
        n += 1