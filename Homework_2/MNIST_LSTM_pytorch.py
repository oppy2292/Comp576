"""
Created on Wed Oct 20 20:40 2020

@author: Yashwanth Lagisetty
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import torch.nn.functional as F


def import_data(root, batch_size=50):
    """
    This function loads MNIST data
    :param root: directory of the MNIST data
    :param batch_size: batch size for data loader
    """
    # Create transformation for MNIST data. Normalize is performed per:
    # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])

    # Load train and test sets
    train_set = datasets.MNIST(root=root, download=True, train=True, transform=transform)
    test_set = datasets.MNIST(root=root, download=True, train=False, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_device():
    """
    Function designates GPU as computational device if cuda is available else defaults to CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class RNN(nn.Module):
    def __init__(self, nHidden, nClasses, nInput, type='rnn'):
        super(RNN, self).__init__()
        self.nHidden = nHidden
        self.nClasses = nClasses
        self.nInput = nInput
        self.type = type
        if type == 'rnn':
            self.rnn = nn.RNN(input_size=self.nInput, hidden_size=self.nHidden, num_layers=1, nonlinearity='tanh',
                              batch_first=True)
        elif type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.nInput, hidden_size=self.nHidden, num_layers=1, batch_first=True)
        elif type == 'gru':
            self.rnn = nn.GRU(input_size=self.nInput, hidden_size=self.nHidden, num_layers=1, batch_first=True)
        else:
            raise KeyError('Type not supported, please choose type = rnn, lstm or gru')
        self.fc = nn.Linear(in_features=self.nHidden, out_features=self.nClasses)
        # nn.init.normal_(self.fc.weight)
        # nn.init.normal_(self.fc.bias)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def train(model, train_loader, test_loader, epochs, writer, device, nSteps, nInput, optimizer, lossF):
    """
    Train the function
    @param model: RNN model
    @param train_loader: data loader for training data
    @param test_loader: data loader for testing data
    @param epochs: number of epochs to train
    @param writer: summary writer for recording analytics
    @param device: device on which to compute
    @param nInput: number of input for sequence, for MNIST 28
    @param nSteps: number of sequence steps, for MNIST if input is 28, then nSteps = 28
    @param optimizer: training methodology, e.g. ADAM, SGD, etc
    @param lossF: loss function
    """
    iter_counter = 1
    for epoch in range(epochs):
        for data in train_loader:
            images, labels = data[0].to(device), data[1].to(device)
            images = images.view(-1, nInput, nSteps)

            optimizer.zero_grad()

            prob = model.forward(images)
            loss = lossF(prob, labels)

            loss.backward()
            optimizer.step()

            if iter_counter % 100 == 0:
                _, preds = torch.max(prob, 1)
                acc = (preds == labels) * 1
                acc = acc.type(torch.float)
                acc = torch.mean(acc).item()

                print("At Iteration {}\nTraining Accuracy is {}\nTraining Loss is {}".format(iter_counter, acc,
                                                                                             loss.item()))
                writer.add_scalar("Loss/Train", loss.item(), iter_counter)
                writer.add_scalar("Acc/Train", acc, iter_counter)

            if iter_counter % 500 == 0:
                tot_correct = 0
                total = 0
                for data in test_loader:
                    test_images, test_labels = data[0].to(device), data[1].to(device)
                    test_images = test_images.view(-1, nInput, nSteps)

                    prob = model.forward(test_images)
                    _, preds = torch.max(prob, 1)
                    correct = (preds == test_labels) * 1
                    correct = correct.type(torch.float)
                    correct = torch.sum(correct).item()

                    tot_correct += correct
                    total += len(test_labels)
                test_acc = tot_correct / total
                # print('Testing Accuracy is {}'.format(test_acc))
                writer.add_scalar("Acc/Test", test_acc, iter_counter)

            iter_counter += 1


def main():
    learningRate = 0.001
    batchSize = 50
    nInput = 28  # we want the input to take the 28 pixels
    nSteps = 28  # every 28
    nHidden = 128  # number of neurons for the RNN
    nClasses = 10  # this is MNIST so you know
    type = 'rnn'
    writer_path = './MNIST_results/'+type+'_hidden_units_'+str(nHidden)+'_lr_'+str(learningRate)+'/'

    train_loader, test_loader = import_data(root='../Homework 1/MNIST_data/', batch_size=batchSize)
    device = get_device()
    writer = SummaryWriter(log_dir=writer_path)
    rnn = RNN(nHidden=nHidden, nClasses=nClasses, nInput=nInput, type=type)
    rnn.to(device)

    optimizer = optim.Adam(rnn.parameters(), lr=learningRate)
    lossF = nn.CrossEntropyLoss()

    train(model=rnn, train_loader=train_loader, test_loader=test_loader, epochs=10, writer=writer, device=device,
          nSteps=nSteps, nInput=nInput, optimizer=optimizer, lossF=lossF)

    torch.save(rnn.state_dict(), './model_weights/rnn_model')


if __name__ == "__main__":
    main()
