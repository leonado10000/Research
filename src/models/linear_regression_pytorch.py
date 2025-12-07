import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

x = torch.randn(6, 3, 4)
model = LR(4, 6)

y = torch.randn(6, 3, 6)

mse = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x)
    loss = mse(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# def run_model(input_data):
#     return model(input_data)

# run_model(x)
