"""Linear Regression Model Implementation."""
import random

class Tensor:
    """A simple Tensor"""
    def __init__(self, data):
        self.data = data
        self.grad = None

class LinearRegression:
    """Linear Regression Model"""
    def __init__(self, input_dim, output_dim, bias=False):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # normal weight untransposed. i.e output_dim x input_dim
        self.weights = [
            [Tensor(random.uniform(-1/(output_dim**.5), 1/(output_dim**.5))) 
                for _ in range(input_dim)]
            for _ in range(output_dim)
        ]
        self.bias = Tensor([random.uniform(-1/output_dim**.5, 1/output_dim**.5) for _ in range(output_dim)]) if bias else None

    def __call__(self, X):
        print("shape X:", len(X), len(X[0]), "shape weights:", len(self.weights), len(self.weights[0]))
        y_pred = []
        for x in X:
            out = []
            for j in range(self.output_dim):
                z = sum([x[k] * self.weights[j][k].data for k in range(self.input_dim)])
                out.append(z)
            y_pred.append(out)
        return y_pred

class MSELoss:
    """Mean Squared Error Loss"""
    def __init__(self, model):
        self.loss = 0
        self.model = model

    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = 0
        for i in range(len(y_pred)):
            for j in range(len(y_pred[0])):
                self.loss += (y_pred[i][j] - y_true[i][j]) ** 2
        self.loss = self.loss / (len(y_pred) * len(y_pred[0]))
        return self
    
    def __repr__(self):
        return f"{self.loss}"

    def backward(self, X):
        n = len(self.y_pred)
        m = self.model.output_dim
        for j in range(self.model.output_dim):
            for k in range(self.model.input_dim):
                grad_sum = 0
                for i in range(n):
                    grad_sum += (self.y_pred[i][j] - self.y_true[i][j]) * X[i][k]
                self.model.weights[j][k].grad = (2 / (n * m)) * grad_sum  # ✅ divide by n*m


class SGD:
    """Stochastic Gradient Descent Optimizer"""
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        for i in range(self.model.output_dim):
            for j in range(self.model.input_dim):
                self.model.weights[i][j].data -= self.lr * self.model.weights[i][j].grad

X = [[random.uniform(0,10) for _ in range(3)] for _ in range(4)]
y = [[random.uniform(0,10) for _ in range(4)] for _ in range(4)]

model = LinearRegression(3,4,bias=False)
mse = MSELoss(model)
sgd = SGD(model, lr=0.01)

# training
for epoch in range(100):
    y_pred = model(X)
    loss = mse(y_pred, y)
    loss.backward(X)
    sgd.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")