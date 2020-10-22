"""
Created on Wed Oct 20 20:40 2020

@author: Yashwanth Lagisetty
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import torch.nn.functional as F


def import_data(cifar_path, n_classes, n_train, n_test, imsize, n_channels=1):
    """
    Import CIFAR10 data into train and test sets
    :param cifar_path: Path of CIFAR10 folder
    :param n_classes: Number of CIFAR10 classes
    :param n_train: number of training samples per class
    :param n_test: number of testing samples per class
    :param imsize: size of training and testing images, default = 28
    :param n_channels: number of image channels, default = 1
    """
    Train = np.zeros((n_classes * n_train, n_channels, imsize, imsize))
    Train_labels = np.zeros((n_classes * n_train, n_classes))
    Test = np.zeros((n_classes * n_test, n_channels, imsize, imsize))
    Test_labels = np.zeros((n_classes * n_test, n_classes))

    train_counter = 0
    test_counter = 0

    for cifar in range(n_classes):
        for image in range(n_train):
            path = cifar_path + 'Train/%d/Image%05d.png' % (cifar, image)
            im = plt.imread(path)
            # im = im.astype(float) / 255 #images are already grayscale and normalized so is this necessary?
            Train[train_counter, :, :, :] = im
            Train_labels[train_counter, cifar] = 1
            train_counter += 1

    for cifar in range(n_classes):
        for image in range(n_test):
            path = cifar_path + 'Test/%d/Image%05d.png' % (cifar, image)
            im = plt.imread(path)
            # im = im.astype(float) / 255 #images are already grayscale and normalized so is this necessary?
            Test[test_counter, :, :, :] = im
            Test_labels[test_counter, cifar] = 1
            test_counter += 1

    Train = torch.from_numpy(Train)
    Test = torch.from_numpy(Test)
    Train_labels = torch.from_numpy(Train_labels)
    Test_labels = torch.from_numpy(Test_labels)
    return Train, Train_labels, Test, Test_labels


def get_device():
    """
    Function designates GPU as computational device if cuda is available else defaults to CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class LeNet5(nn.Module):
    """
    LeNet class
    """

    def __init__(self):
        """
        Initialize components of LeNet5
        """
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        # torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        # torch.nn.init.constant_(self.conv1.bias, val=0.1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        # torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        # torch.nn.init.constant_(self.conv2.bias, val=0.1)

        self.FC1 = nn.Linear(in_features=4 * 4 * 64, out_features=1024)
        # torch.nn.init.kaiming_normal_(self.FC1.weight)
        # torch.nn.init.constant_(self.FC1.bias, val=0.1)

        self.FC2 = nn.Linear(in_features=1024, out_features=10)
        # torch.nn.init.kaiming_normal_(self.FC2.weight)
        # torch.nn.init.constant_(self.FC2.bias, val=0.1)

    def forward(self, x):
        """
        Performs Forward pass of LeNet5
        :param x: input data
        """
        self.a1 = F.relu(self.conv1(x))
        x = F.max_pool2d(self.a1, 2)
        self.a2 = F.relu(self.conv2(x))
        x = F.max_pool2d(self.a2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        return F.softmax(x, dim=1)


def train(model, X_train, y_train, X_test, y_test, lossF, optimizer, device, writer, batch_size, epochs=5):
    """
    Trains model
    :param model: model to be trained
    :param X_train: training data
    :param y_train: training labels
    :param batch_size: size of training batches
    :param lossF: pytorch loss function
    :param optimizer: pytorch optimizer
    :param device: device on which to do computations
    :param epochs: number of epochs
    :param writer: summary writer to record training analytics
    """
    iter_counter = 0
    for epoch in range(epochs):
        samp_idx = np.arange(X_train.shape[0])
        np.random.shuffle(samp_idx)
        split_idx = list(range(batch_size, len(samp_idx), batch_size))
        batches = np.array_split(samp_idx, split_idx)
        for batch in batches:
            train_batch = X_train[batch, :, :, :].to(device)
            train_labels = torch.argmax(y_train[batch, :], dim=1).to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Predict labels and calculate loss
            y_probs = model.forward(train_batch.float())
            loss = lossF(y_probs, train_labels.type(torch.long))

            # Perform backpropagation
            loss.backward()
            optimizer.step()

            # Calculate accuracy and loss
            _, y_preds = torch.max(y_probs, 1)
            correct = (y_preds == train_labels.float()) * 1
            correct = correct.type(torch.float)
            acc = torch.mean(correct).item()

            if iter_counter % 100 == 0:
                print("Loss is :{}".format(loss.item()))
                optimizer.zero_grad()
                test_idx = np.arange(X_test.shape[0])
                np.random.shuffle(test_idx)
                split_idx = list(range(batch_size, len(test_idx), batch_size))
                test_batches = np.array_split(test_idx, split_idx)
                num_correct = 0
                for batch in test_batches:
                    test_batch = X_test[batch, :, :, :].to(device)
                    test_label = torch.argmax(y_test[batch, :], dim=1).to(device)

                    y_probs = model.forward(test_batch.float())
                    _, y_preds = torch.max(y_probs, 1)
                    correct = (y_preds == test_label.float()) * 1
                    correct = correct.type(torch.float)
                    correct = torch.sum(correct).item()
                    num_correct += correct
                test_acc = num_correct / len(y_test)
                writer.add_scalar("Acc/Test", test_acc, iter_counter)
                writer.add_scalar("Loss/Train", loss.item(), iter_counter)
                writer.add_scalar("Acc/Train", acc, iter_counter)
                writer.add_scalar("mean_W1", torch.mean(model.conv1.weight).item(), iter_counter)
                writer.add_scalar("mean_W2", torch.mean(model.conv2.weight).item(), iter_counter)
                writer.add_scalar("mean_act1", torch.mean(model.a1).item(), iter_counter)
                writer.add_scalar("mean_act2", torch.mean(model.a2).item(), iter_counter)

                print('At step {}:\nTraining Loss is {}\nTraining Accuracy is {}\nTesting Accuracy is {}'.format(
                    iter_counter, loss.item(), acc, test_acc))

            iter_counter += 1


def plot_Gabor(model, model_weights_path):
    """
    Function plots the filters from the fist convolution layer
    :param model: LeNet model
    :param model_weights_path: path to saved model weights
    :return:
    """
    model.load_state_dict(torch.load(model_weights_path))
    filters = model.conv1.weight.data.numpy()

    plt.figure(figsize=(10, 10))
    for i in range(filters.shape[0]):
        plt.subplot(6, 6, i + 1)
        plt.imshow(filters[i, 0, :, :], cmap='gray')
    plt.show()


def activation_stats_test(model, test_images, test_labels, writer, batch_size, device, model_weights_path):
    """
    Computes the statistics for trained model's activations on test images 
    :param model: LeNet Model
    :param test_images: test set images
    :param test_labels: test set image labels
    :param writer: summary writer to record activations
    :param batch_size: batch size 
    :param device: device on which to run computations
    :param model_weights_path: path for model weights
    """
    model.load_state_dict(torch.load(model_weights_path))
    idx = np.arange(test_images.shape[0])
    np.random.shuffle(idx)
    split_idx = list(range(batch_size, len(idx), batch_size))
    test_batches = np.array_split(idx, split_idx)
    num_correct = 0
    iter_counter = 1
    for batch in test_batches:
        test_batch = test_images[batch, :, :, :].to(device)
        test_label = torch.argmax(test_labels[batch, :], dim=1).to(device)

        y_probs = model.forward(test_batch.float())
        _, y_preds = torch.max(y_probs, 1)
        correct = (y_preds == test_label.float()) * 1
        correct = correct.type(torch.float)
        correct = torch.sum(correct).item()
        num_correct += correct

        writer.add_scalar("mean_act1", torch.mean(model.a1).item(), iter_counter)
        writer.add_scalar("mean_act2", torch.mean(model.a2).item(), iter_counter)
        writer.add_scalar("std_act1", torch.std(model.a1).item(), iter_counter)
        writer.add_scalar("std_act2", torch.std(model.a2).item(), iter_counter)
        writer.add_scalar("max_act1", torch.max(model.a1).item(), iter_counter)
        writer.add_scalar("max_act2", torch.max(model.a2).item(), iter_counter)
        writer.add_scalar("min_act1", torch.min(model.a1).item(), iter_counter)
        writer.add_scalar("min_act2", torch.min(model.a2).item(), iter_counter)

        writer.add_histogram('Conv1 Activation', model.a1, iter_counter)
        writer.add_histogram('Conv2 Activation', model.a2, iter_counter)
        iter_counter += 1
    test_acc = num_correct / len(test_labels)
    print('Test accuracy is {}'.format(test_acc))


def main():
    X_train, y_train, X_test, y_test = import_data(cifar_path='./CIFAR10/', n_classes=10, n_train=1000, n_test=100,
                                                   imsize=28)
    writer = SummaryWriter(log_dir='./CIFAR10_results')
    device = get_device()
    net = LeNet5()
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=2e-4, weight_decay=0.0001)
    lossF = nn.CrossEntropyLoss()

    train(model=net, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, batch_size=32, lossF=lossF,
          optimizer=optimizer, epochs=200, device=device, writer=writer)

    torch.save(net.state_dict(), './model_weights/net_model')

    # If Gabor filter visualization is desired:
    # plot_Gabor(net,'./model_weights/net_model')

    # If statistics on activation layer on test images is desired:
    # activation_stats_test(model=net, test_images=X_test, test_labels=y_test, writer=writer, batch_size=32,
    #                       device=device, model_weights_path='./model_weights/net_model')

if __name__ == "__main__":
    main()
