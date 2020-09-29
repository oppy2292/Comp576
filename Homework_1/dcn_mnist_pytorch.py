import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F


def import_data(root, batch_size):
    # Create transformation for MNIST data. ToTensor converts values to 0-1, Normalize per 0.1307 and 0.3081 which are
    # mean and std of MNIST: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])

    # Load train and test sets
    train_set = datasets.MNIST(root=root, download=True, train=True, transform=trans)
    test_set = datasets.MNIST(root=root, download=True, train=False, transform=trans)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class DCN(nn.Module):
    def __init__(self):
        super(DCN, self).__init__()
        # We want architecture that is as follows:
        # conv1(5-5-1-32) -> ReLU -> maxpool(2-2) -> conv2(5-5-32-64) -> ReLU -> maxpool(2-2)
        # -> FC(1024) -> ReLU -> Dropout(0.5) -> softmax(10)

        # Initialize the operation and weights, biases
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        # torch.nn.init.normal_(self.conv1.weight)
        # torch.nn.init.constant_(self.conv1.bias, val=0.1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        # torch.nn.init.normal_(self.conv2.weight)
        # torch.nn.init.constant_(self.conv2.bias, val=0.1)

        self.FC1 = nn.Linear(in_features=64 * 4 * 4, out_features=1024)
        self.drop = nn.Dropout(p=0.5)
        self.FC2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.FC1(x))
        x = self.drop(x)
        x = self.FC2(x)
        return F.softmax(x)


def train_model(model, device, train_loader, epochs, optimizer, lossF):
    for epoch in range(epochs):
        for idx, data in enumerate(train_loader):
            imgs, labels = data[0].to(device), data[1].to(device)

            # We wanto to make sure all gradient parameters are set to 0 before we calculate again
            optimizer.zero_grad()

            # Perform Forward pass and calculalate loss
            y_probs = model(imgs)
            loss = lossF(y_probs, labels)

            # Perform backpropagation
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                _,preds = torch.max(y_probs,1)
                acc = (preds == labels) * 1
                acc = acc.type(torch.float)
                acc = torch.mean(acc)
                print('At iter = {}\nTraining Accuracy = {}\nTraining loss = {}'.format(idx, acc, loss.item()))

            if idx % 1100 == 0:
                _,preds = torch.max(y_probs,1)
                acc = (preds == labels) * 1
                acc = acc.type(torch.float)
                acc = torch.mean(acc)
                print('AT EPOCH {}\nTraining Accuracy = {}\nTraining loss = {}'.format(epoch+1, acc, loss.item()))


def test_model(model, test_loader, device,lossF):
    num_correct = 0
    sum_loss = 0
    total_samps = 0
    for data in test_loader:
        imgs, labels = data[0].to(device), data[1].to(device)
        y_probs = model(imgs)
        loss = lossF(y_probs, labels)

        _, preds = torch.max(y_probs, 1)
        count = (preds == labels) * 1
        count = count.type(torch.float)
        count = torch.sum(count).item()

        num_correct += count
        total_samps += len(labels)
        sum_loss += loss.item()

    acc = num_correct / total_samps
    avg_loss = sum_loss / total_samps
    print('Testing set accuracy is {}\nTesting set loss is {}'.format(acc, avg_loss))


def main():
    train_loader, test_loader = import_data(root='./MNIST_data', batch_size=50)
    device = get_device()
    dcn = DCN()
    dcn.to(device)

    optimizer = optim.Adam(dcn.parameters(),lr=0.001)
    lossF = nn.CrossEntropyLoss()

    train_model(dcn,train_loader=train_loader,epochs=5,optimizer=optimizer,lossF=lossF,device=device)
    test_model(dcn,test_loader=test_loader,device=device,lossF=lossF)

if __name__ == "__main__":
    main()
