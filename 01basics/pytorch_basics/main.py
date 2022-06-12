from cgitb import reset
from random import shuffle
from cv2 import RQDecomp3x3
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w * x + b
y.backward()

print(x.grad)  # x.grad = 2
print(w.grad)  # w.grad = 1
print(b.grad)  # b.grad = 1
print('-'*10)
# 1. basic autograd example

x = torch.randn(10, 3)
y = torch.randn(10, 2)

# build a fully connected layer
linear = nn.Linear(3, 2)
print('w:', linear.weight)
print('b:', linear.bias)

# 2. basic autograd example 2
# build loss function and optimazer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forwad pass
pred = linear(x)

# compute loss
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass
loss.backward()

# print out grads
print('dL/dw:', linear.weight.grad)
print('dL/db:', linear.bias.grad)

# 1-step gradient descent
optimizer.step()

# print out loss after 1-step gradient descent
pred = linear(x)
loss = criterion(pred, y)
print('losss after 1 step optimization:', loss.item())

# 3. load data from numpy

# create a numpy array
x = np.array([[1, 2], [3, 4]])
print("shape of x: ", x.shape)

# convert the torch tensor to a numpy array
y = torch.from_numpy(x)

# convert the torch tensor to a numpy array
z = y.numpy()

# 4. input pipeline
# download and construct CIFART-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='../../data/', train=True, transform=transforms.ToTensor(), download=True)
image, label = train_dataset[0]
print(image.size())
print(label)

# data loader
train_loader = torch.utils.Dataloader(
    dataset=train_dataset, batch_size=64, shuffle=True)

data_iter = iter(train_loader)
images, labels = data_iter.next()

for imgs, labels in train_loader:
    # train code should be written here
    pass

# 5. input pipeline for custom dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        # TODO
        # 1. Initialize file path or list of file names
        pass

    def __getitem__(self, index):
        # TODO
        # 1. read one data from file(e.g. using numpy fromfile, PIL.Image.open)
        # 2. preprocess the data (e.g. torchvision.Transform)
        # 3. return a data pair (e.g. image and label)
        pass

    def __len__(self):
        # return size of the dataset
        pass


custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True))


#  6.pretrained model 
resnet = torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100) # 100 is an example

imgs = torch.randn(64, 3, 224, 224)
outputs = resnet(imgs)
print(outputs.size())

# 7. save and load model  
# save and load entire model
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')
# save and load only the model parameters(recomendded)
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))