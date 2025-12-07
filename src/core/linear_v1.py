import random
import sys
sys.path.append("")
from core.tensor_v1 import Tensor
from src.core.meanSquareError import MSE

class Linear:
    def __init__(self, input_size, output_size, lr=.01):
        self.name = "Linear("+str(input_size)+"x"+str(output_size)+")"
        self.input_size = input_size
        self.output_size = output_size
        self.weights = Tensor().create_tensor(shape=[input_size, output_size], type="random", origin=self, parent=None)
        self.bias = random.uniform(-1/input_size, -1/input_size)
        self.lr = lr
        
    def forward(self, X:Tensor) -> Tensor:
        return Tensor(X.dot(self.weights).data, origin=self, parent_tensor=X)
    
    # sample here; 
    # problem is the following is gradient change for MSE loss, 
    # so this applies to all weight changes which are MSE
    # Non MSE IDK
    # We implement the gradient with the Loss class and apply it for each model ig.  
    # def _gradient_change(self, y_true, y_pred, X):
    #     n = len(y_true)
    #     m = len(y_true[0])
    #     for j in range(m):
    #         for i in range(n):
    #             self.weights.data[i][j] -= self.lr * (2 / (n * m)) * (y_pred.data[i][j] - y_true.data[i][j]) * X.data[i][j]
    #     print("Gradient changed")

# correct output 
y = Tensor().create_tensor([1,10], type=1)
input_tensor = Tensor().create_tensor([1,784], type=1)
ll1 = Linear(784, 10)
ll2 = Linear(10, 10)
mse = MSE()



# # training
mapping = {}
for epoch in range(10):
    out1 = ll1.forward(input_tensor)
    out2 = ll2.forward(out1)
    loss = mse(out2, y)
    # print(out2.data)
    print(f"epoch = {epoch}, loss = {loss.item()}")
    loss.backward()

min_losses = sorted(mapping.keys())
for loss in min_losses[:10]:
    print(f"epoch {mapping[loss]}, loss = {loss}")