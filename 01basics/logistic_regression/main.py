from pickletools import optimize
from random import shuffle
from xml.etree.ElementPath import prepare_predicate
from xml.sax.xmlreader import InputSource
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

input_size = 28 * 28    # 784
num_classes = 10
num_epochs = 5
batch_size = 100
lr = 0.001

train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = nn.Linear(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.reshape(-1, input_size)

        # forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        if(i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# test model 
with torch.no_grad():
    correct = 0
    total = 0
    for i, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.reshape(-1, input_size)
        pred = model(imgs)
        # _, pred = torch.max(pred.data, 1)
        pred = torch.argmax(pred.data,1)
        total += labels.size(0)
        correct += (pred==labels).sum()
    print('accuracy of model on the 1000 test imgs:{}'.format(100*correct/total))


torch.save(model.state_dict(), 'model.ckpt')


