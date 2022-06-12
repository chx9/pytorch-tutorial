from statistics import mode
import torch
import numpy as np


# simple linear regression using pytorch
# with out nn.Module

# data generate
w = 3 # true weight
b = 1 # true bias
x_ = np.array([np.linspace(0, 3, 50)])
x_ = x_.reshape(-1, 1)   # reshape for batch
noise = np.random.normal(0, 0.1, (50, 1))
y_ = x_*w+b+noise
y_ = y_.reshape(-1, 1)   # reshape for batch


weight = torch.tensor(0.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)
criterion = torch.nn.MSELoss()
lr = 0.01
epochs = 1000
optimizer = torch.optim.SGD([weight, bias], lr)

for epoch in range(epochs+1):
    # remember clear optimazer's grad 
    optimizer.zero_grad()
    x_train = torch.from_numpy(x_).type(torch.float32)
    y_train = torch.from_numpy(y_).type(torch.float32)

    pred = weight*x_train + bias # model preds
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
            print("epoch: {}, loss {}".format(epoch, loss.item()))
            print("w: {}, b: {}".format(weight.item(), bias.item()))








# x_values = [i for i in range(11)]
# x_train = np.array(x_values, dtype=np.float32)
# x_train = x_train.reshape(-1, 1)
# print(x_train.shape)
# y_values = [2*i + 1 for i in x_values]
# y_train = np.array(y_values, dtype=np.float32)
# y_train = y_train.reshape(-1, 1)


# class LinearRegression(torch.nn.Module):
#     def __init__(self, input_size, output_size) -> None:
#         super().__init__()
#         self.linear = torch.nn.Linear(input_size, output_size)

#     def forward(self, x):
#         out = self.linear(x)
#         return out
        

# input_dim = 1
# output_dim = 1
# lr = 0.01
# epochs = 1000
# model = LinearRegression(input_dim, output_dim)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr)


# for epoch in range(epochs):
#     input = torch.tensor(torch.from_numpy(x_train)) 
#     labels = torch.tensor(torch.from_numpy(y_train))
#     optimizer.zero_grad()
#     pred = model(input)

#     loss = criterion(labels, pred)
#     loss.backward()
#     optimizer.step()
#     if epoch % 500 == 0:
#         print('epoch {}, loss {}'.format(epoch, loss.item()))
#         print(list(model.parameters()))
