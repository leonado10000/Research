import sys
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as tv
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

sys.path.append("./src")
from core.convolution import Conv2D
from core.pooling import Pool
from core.linear_v1 import Linear
from core.relu_v1 import relu

class tensor:
    def __init__(self, X):
        self.value = X
        self.shape = []
        temp = X
        while True:
            if type(temp) != list:
                break
            self.shape.append(len(temp))
            temp = temp[0]


class cnn:
    def __init__(self):
        self.model_name = "cnn_scratch"
        self.conv1 = Conv2D(1, 6, 5)
        self.pool = Pool(2, 2, "max")
        self.conv2 = Conv2D(6, 16, 5)

        self.fc1 = Linear(16*4*4, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 16)

    def output_shape(self, X):
        image = X
        h,w = image.shape[-2],image.shape[-1]
        k_h, k_w = self.kernel.shape[-2],self.kernel.shape[-1]
        
        h_out = (h-k_h-2*self.padding)//self.stride[0] +1
        w_out = (w-k_w-2*self.padding)//self.stride[1] +1
        return h_out,w_out
    
    def forward(self, X):
        x = self.pool.forward((relu(self.conv1.forward(X))))
        x = self.pool.forward((relu(self.conv2.forward(x))))

        # print("Shape after conv and pooling layers: ", x.shape)
        x = x.reshape(-1, 16*4*4)
        x = relu(self.fc1.forward(x))
        x = relu(self.fc2.forward(x))
        x = self.fc3.forward(x)
        return x

X = tensor(
    [[1, 2, 3, 4], 
     [5, 6, 7, 8], 
     [9, 10, 11, 12],
     [13, 14, 15, 16]])

# feature_map = np.array([
#     [   # first image
#         [
#             [1, 2, 3, 4, 5, 6, 7],
#             [8, 9, 10, 11, 12, 13, 14],
#             [15, 16, 17, 18, 19, 20, 21],
#             [22, 23, 24, 25, 26, 27, 28],
#             [29, 30, 31, 32, 33, 34, 35],
#             [36, 37, 38, 39, 40, 41, 42],
#             [43, 44, 45, 46, 47, 48, 49]
#         ]
#     ],
#     [   # second image
#         [
#             [49, 48, 47, 46, 45, 44, 43],
#             [42, 41, 40, 39, 38, 37, 36],
#             [35, 34, 33, 32, 31, 30, 29],
#             [28, 27, 26, 25, 24, 23, 22],
#             [21, 20, 19, 18, 17, 16, 15],
#             [14, 13, 12, 11, 10, 9, 8],
#             [7, 6, 5, 4, 3, 2, 1]
#         ]
#     ]
# ])

dataset = tv.MNIST("./src/data/", train=True, transform=transforms.ToTensor(), download=False)
trainloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

model = cnn()
loss_fn = nn.CrossEntropyLoss()
# print(model.forward(feature_map))

running_loss_list = []
while True:
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # print("inputs shape ====> ", inputs.shape, "labels shape ====> ", labels.shape)
            outputs = model.forward(inputs)
            loss = loss_fn(torch.as_tensor(outputs), labels)
            print(f"Loss ({i+1}/ {len(trainloader)}) ====> {loss.item()}")
            running_loss += loss.item()
        running_loss_list.append(running_loss)