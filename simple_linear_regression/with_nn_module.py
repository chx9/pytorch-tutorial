from pickletools import optimize
from matplotlib.lines import Line2D
import torch
import numpy as np
x = np.linspace(1, 5, 100)
x = x.reshape(-1, 1)
noise = np.random.normal(0, 0.1, (100, 1))
y = x*3 + 1 + noise


model = torch.nn.Linear(1, 1)
epochs = 1000
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr)
criterion = torch.nn.MSELoss()

for epoch in range(epochs):
    optimizer.zero_grad()
    x_train = torch.from_numpy(x).type(torch.float32)
    y_train = torch.from_numpy(y).type(torch.float32)
    pred = model(x_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss.item()))

        print("w: {}, b: {}".format(model.weight.item(), model.bias.item()))