import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.datasets as tv
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import datetime as dt
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model_name = "cnn_pytorch"
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 16)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

dataset = tv.MNIST("./src/data/", train=True, transform=transforms.ToTensor(), download=False)
trainloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)
model_save_path = f"""./artifacts/exported/cnn/{device}_pytorch_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"""
# print(f"Model will be saved to: {model_save_path}")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
criterion = model.loss_function
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

running_loss_list = []
train_start = time.time()
while True:
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i >= 1000:
                break
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            print("shapes",outputs.shape, labels.shape)
            print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 800 == 799:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 800)
                    )
                running_loss_list.append(running_loss)
                running_loss = 0.0
#     print('Finished Training')
#     torch.save(model.state_dict(), model_save_path)
#     break

# train_end = time.time()
# print(f"Training time: {train_end - train_start:.2f} seconds")

# eval_start = time.time()
# eval_model = Net()
# eval_model.to(device)
# eval_model.load_state_dict(torch.load(model_save_path))
# dataset = tv.MNIST("./src/data/", train=False, transform=transforms.ToTensor(), download=True)
# testloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False)

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = eval_model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# eval_end = time.time()
# print(f"Evaluation time: {eval_end - eval_start:.2f} seconds")

# accuracy = 100 * correct / total
# print("\nAccuracy evaluation:")
# print(f"Accuracy of the model on the 10,000 MNIST test images: {accuracy:.2f}%")
# print('Evaluation Finished')
