import random
import math
import sys
sys.path.append("")
from src.core.tensor import Tensor
from src.core.utility import is_scalar, shape_of, flatten, unflatten, elementwise_op, elementwise_map, zeros_like, add_inplace, sub_inplace, mul_inplace, scale, sum_all
from src.core.meanSquareError import MSE

class Linear:
    def __init__(self, in_dim, out_dim, lr=0.01, init_scale=None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        # if no weight limit value given, using Xaiver Normal 
        if not init_scale:
            init_scale = math.sqrt(6/(in_dim))
        # weights: (in_dim x out_dim)
        w_data = [[random.uniform(-init_scale, init_scale) for _ in range(out_dim)] for _ in range(in_dim)]
        b_data = [0.0 for _ in range(out_dim)]
        self.weights = Tensor(w_data, op="weights")
        self.bias = Tensor(b_data, op="bias")

    def __repr__(self):
        return f"Linear ({self.in_dim}x{self.out_dim})"

    def forward(self, x: Tensor):
        # x: (1 x in_dim) or (m x in_dim) -- here we assume (1 x in_dim) for your usage
        # x.matmul(weights) : (m x in_dim) @ (in_dim x out_dim) -> (m x out_dim)
        out = x.matmul(self.weights)
        # add bias (row-wise)
        def add_bias(mat, bias):
            return [[mat[i][j] + bias[j] for j in range(len(bias))] for i in range(len(mat))]
        out_data = add_bias(out.data, self.bias.data)
        out_tensor = Tensor(out_data, parents=[x, self.weights, self.bias], op="linear")
        def _backward():
            if out_tensor.grad is None:
                return
            g = out_tensor.grad  # m x out_dim
            # dW = x.T @ g  where x: m x in_dim -> x.T: in_dim x m ; g: m x out_dim
            m = x.shape[0]
            in_d = self.in_dim
            out_d = self.out_dim
            # compute dW (in_dim x out_dim)
            dW = [[0.0 for _ in range(out_d)] for _ in range(in_d)]
            for i in range(in_d):
                for j in range(out_d):
                    s = 0.0
                    for k in range(m):
                        s += x.data[k][i] * g[k][j]
                    dW[i][j] = s
            # compute db (out_dim) : sum over m
            db = [sum(g[k][j] for k in range(m)) for j in range(out_d)]
            # compute dx = g @ W.T  (m x out_dim) @ (out_dim x in_dim) = m x in_dim
            dx = [[0.0 for _ in range(in_d)] for _ in range(m)]
            for r in range(m):
                for c in range(in_d):
                    s = 0.0
                    for j in range(out_d):
                        s += g[r][j] * self.weights.data[c][j]
                    dx[r][c] = s

            # accumulate grads into weight and bias tensors
            self.weights.grad = add_inplace(self.weights.grad if self.weights.grad else zeros_like(self.weights.data), dW)
            self.bias.grad = add_inplace(self.bias.grad if self.bias.grad else zeros_like(self.bias.data), db)

            # push gradient to input tensor x
            x.grad = add_inplace(x.grad if x.grad else zeros_like(x.data), dx)

        out_tensor._backward = _backward
        return out_tensor

    # simple parameter update method (call after backward)
    def step(self):
        if self.weights.grad is None:
            return
        # gradient descent update
        self.weights.data = sub_inplace(self.weights.data, scale(self.weights.grad, self.lr))
        # reset grads on parameters
        self.weights.grad = None
        self.bias.data = sub_inplace(self.bias.data, scale(self.bias.grad if self.bias.grad else zeros_like(self.bias.data), self.lr))
        self.bias.grad = None

# # correct output 
# y = Tensor.from_shape([1,10], value=1)
# input_tensor = Tensor.from_shape([1,784], value=2)
# ll1 = Linear(784, 128)
# ll2 = Linear(128, 10)
# mse = MSE()



# # # training
# mapping = {}
# for epoch in range(10):
#     out1 = ll1.forward(input_tensor).relu()
#     out2 = ll2.forward(out1)
#     loss = mse(out2, y)
#     print(f"epoch = {epoch}, loss = {loss.item()}")
#     loss.backward()
#     ll1.step()
#     ll2.step()

# min_losses = sorted(mapping.keys())
# for loss in min_losses[:10]:
#     print(f"epoch {mapping[loss]}, loss = {loss}")