import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import torch.nn.functional as F


def import_data(root, batch_size):
    """
    This function imports the MNIST data and loads them into dataloaders
    :param root: the root directory of the MNIST data or location to download them
    :param batch_size: Batch size for the data loader
    :return: train and test data loaders
    """
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
    """
    Function designates GPU as computational device if cuda is available else defaults to CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class DCN(nn.Module):
    """
    Neural Network Class
    """
    def __init__(self):
        """
        Initialize the necessary components
        """
        super(DCN, self).__init__()
        # We want architecture that is as follows:
        # conv1(5-5-1-32) -> ReLU -> maxpool(2-2) -> conv2(5-5-32-64) -> ReLU -> maxpool(2-2)
        # -> FC(1024) -> ReLU -> Dropout(0.5) -> softmax(10)

        # Initialize the operation and weights, biases
        # Weights and biases can be easily altered by using a different torch.nn.init method, e.g. torch.nn.init.normal_()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.constant_(self.conv1.bias, val=0.1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.constant_(self.conv2.bias, val=0.1)

        self.FC1 = nn.Linear(in_features=64 * 4 * 4, out_features=1024)
        torch.nn.init.kaiming_normal_(self.FC1.weight)
        torch.nn.init.constant_(self.FC1.bias, val=0.1)

        self.drop = nn.Dropout(p=0.5)
        self.FC2 = nn.Linear(in_features=1024, out_features=10)
        torch.nn.init.kaiming_normal_(self.FC2.weight)
        torch.nn.init.constant_(self.FC2.bias, val=0.1)

    def forward(self, x):
        """
        Performs Forward Pass
        :param x: input
        :return: softmax probabilities for classes
        """
        # We decide to record the activations, and maxpooling outputs in self objects because
        # this will make it easier to call and track the per layer outputs when recording model
        # training analytics
        self.act1 = F.relu(self.conv1(x))
        self.max1 = F.max_pool2d(self.act1, 2)
        self.act2 = F.relu(self.conv2(self.max1))
        self.max2 = F.max_pool2d(self.act2, 2)
        x = self.max2.view(-1, 64 * 4 * 4)
        x = F.relu(self.FC1(x))
        x = self.drop(x)
        x = self.FC2(x)
        return F.softmax(x, dim=1)


def train_model(model, device, train_loader, epochs, optimizer, lossF, writer, test_loader, record_val_test=False):
    """
    This function trains a neural network model
    :param model: Neural network model
    :param device: device on which to perform computations, e.g. GPU vs CPU
    :param train_loader: dataloader for training data
    :param epochs: int for number of epochs
    :param optimizer: pytorch optimizer
    :param lossF: pytorch loss function or self defined loss function
    :param writer: summary writer to record log of training analytics
    :param test_loader: data loader for test set - while this is training loop, homework calls for record of training
                        loss across epochs
    :param record_val_test: boolean designating whether test and validation accuracies should be recorded
    """
    for epoch in range(epochs):
        for idx, data in enumerate(train_loader):
            imgs, labels = data[0].to(device), data[1].to(device)

            # If record_val_test is True, then we will make a validation set which is approx. 10% of the training set
            if record_val_test:
                val_imgs = imgs[45:]
                val_labels = labels[45:]
                imgs = imgs[0:45]
                labels = labels[0:45]

            # We wanto to make sure all gradient parameters are set to 0 before we calculate again
            optimizer.zero_grad()

            # Perform Forward pass and calculalate loss
            y_probs = model(imgs)
            loss = lossF(y_probs, labels)

            # Perform backpropagation
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                _, preds = torch.max(y_probs, 1)
                acc = (preds == labels) * 1
                acc = acc.type(torch.float)
                acc = torch.mean(acc)
                print('At iter = {}\nTraining Accuracy = {}\nTraining loss = {}'.format(idx, acc, loss.item()))

                # Huge list of the things we want to track on TBoard
                # First we keep track of scalars
                global_step = idx + epoch * 1100
                writer.add_scalar("Loss/Train", loss.item(), global_step)
                writer.add_scalar("Acc/Train", acc, global_step)
                writer.add_scalar("max_W1", torch.max(model.conv1.weight).item(), global_step)
                writer.add_scalar("min_W1", torch.min(model.conv1.weight).item(), global_step)
                writer.add_scalar("mean_W1", torch.mean(model.conv1.weight).item(), global_step)
                writer.add_scalar("std_W1", torch.std(model.conv1.weight).item(), global_step)
                writer.add_scalar("max_W2", torch.max(model.conv2.weight).item(), global_step)
                writer.add_scalar("min_W2", torch.min(model.conv2.weight).item(), global_step)
                writer.add_scalar("std_W2", torch.std(model.conv2.weight).item(), global_step)
                writer.add_scalar("mean_W2", torch.mean(model.conv2.weight).item(), global_step)
                writer.add_scalar("max_b1", torch.max(model.conv1.bias).item(), global_step)
                writer.add_scalar("min_b1", torch.min(model.conv1.bias).item(), global_step)
                writer.add_scalar("std_b1", torch.std(model.conv1.bias).item(), global_step)
                writer.add_scalar("mean_b1", torch.mean(model.conv1.bias).item(), global_step)
                writer.add_scalar("max_b2", torch.max(model.conv2.bias).item(), global_step)
                writer.add_scalar("min_b2", torch.min(model.conv2.bias).item(), global_step)
                writer.add_scalar("std_b2", torch.std(model.conv2.bias).item(), global_step)
                writer.add_scalar("mean_b2", torch.mean(model.conv2.bias).item(), global_step)
                writer.add_scalar("Layer 1 act max", torch.max(model.act1).item(), global_step)
                writer.add_scalar("Layer 1 act min", torch.min(model.act1).item(), global_step)
                writer.add_scalar("Layer 1 act std", torch.std(model.act1).item(), global_step)
                writer.add_scalar("Layer 1 act mean", torch.mean(model.act1).item(), global_step)
                writer.add_scalar("Layer 2 act max", torch.max(model.act2).item(), global_step)
                writer.add_scalar("Layer 2 act min", torch.min(model.act2).item(), global_step)
                writer.add_scalar("Layer 2 act std", torch.std(model.act2).item(), global_step)
                writer.add_scalar("Layer 2 act mean", torch.mean(model.act2).item(), global_step)
                writer.add_scalar("Layer 1 MaxPool max", torch.max(model.max1).item(), global_step)
                writer.add_scalar("Layer 1 MaxPool min", torch.min(model.max1).item(), global_step)
                writer.add_scalar("Layer 1 MaxPool std", torch.std(model.max1).item(), global_step)
                writer.add_scalar("Layer 1 MaxPool mean", torch.mean(model.max1).item(), global_step)
                writer.add_scalar("Layer 2 MaxPool max", torch.max(model.max2).item(), global_step)
                writer.add_scalar("Layer 2 MaxPool min", torch.min(model.max2).item(), global_step)
                writer.add_scalar("Layer 2 MaxPool std", torch.std(model.max2).item(), global_step)
                writer.add_scalar("Layer 2 MaxPool mean", torch.mean(model.max2).item(), global_step)

                # Now record Histograms
                writer.add_histogram("W1", model.conv1.weight, global_step)
                writer.add_histogram("W2", model.conv2.weight, global_step)
                writer.add_histogram("b1", model.conv1.bias, global_step)
                writer.add_histogram("b2", model.conv2.bias, global_step)
                writer.add_histogram("Layer1 act", model.act1, global_step)
                writer.add_histogram("Layer2 act", model.act2, global_step)
                writer.add_histogram("Layer1 maxpool", model.max1, global_step)
                writer.add_histogram("Layer2 maxpool", model.max2, global_step)

            if idx % 1100 == 0:
                _, preds = torch.max(y_probs, 1)
                acc = (preds == labels) * 1
                acc = acc.type(torch.float)
                acc = torch.mean(acc)
                print('AT EPOCH {}\nTraining Accuracy = {}\nTraining loss = {}'.format(epoch + 1, acc, loss.item()))

                # If we want to record the validation and test losses at each epoch
                if record_val_test:
                    # Calculate validation accuracy and loss
                    val_preds = model.forward(val_imgs)
                    val_loss = lossF(val_preds, val_labels).item()
                    _, val_preds = torch.max(val_preds, 1)
                    val_acc = (val_preds == val_labels) * 1
                    val_acc = val_acc.type(torch.float)
                    val_acc = torch.mean(val_acc).item()

                    # Calculate training loss and accuracy
                    test_acc, test_loss = test_model(model=model, test_loader=test_loader, device=device, lossF=lossF,
                                                     print_loss_acc=False)
                    writer.add_scalar("Loss/Test", test_loss, epoch)
                    writer.add_scalar("Acc/Test", test_acc, epoch)
                    writer.add_scalar("Loss/Val", val_loss, epoch)
                    writer.add_scalar("Acc/Val", val_acc, epoch)


def test_model(model, test_loader, device, lossF, print_loss_acc=True):
    """
    This function tests a trained model on test dataset
    :param model: trained model
    :param test_loader: dataloader for test set
    :param device: device to run computations, e.g. CPU vs GPU
    :param print_loss_acc: Boolean whether or not to print total test accuracy and average test loss
    :return: total test accuracy and average test loss
    """

    # Keep track of how many are correct, the total number of samples and a list of the losses at each iteration
    num_correct = 0
    losses = []
    total_samps = 0
    for data in test_loader:
        imgs, labels = data[0].to(device), data[1].to(device)

        # Perform forward pass and calculate loss
        y_probs = model(imgs)
        loss = lossF(y_probs, labels)

        # Calculate the number of correct predictions
        _, preds = torch.max(y_probs, 1)
        count = (preds == labels) * 1
        count = count.type(torch.float)
        count = torch.sum(count).item()

        # Update num_correct with the number of correct predictions
        # Update the total number of samples by the number of samples in this batch
        num_correct += count
        total_samps += len(labels)
        losses.append(loss.item())

    acc = num_correct / total_samps
    avg_loss = np.mean(losses)
    if print_loss_acc:
        print('Testing set accuracy is {}\nTesting set loss is {}'.format(acc, avg_loss))
    return acc, avg_loss


def main():
    train_loader, test_loader = import_data(root='./MNIST_data', batch_size=50)
    writer = SummaryWriter(log_dir='./MNIST_results')
    device = get_device()
    dcn = DCN()
    dcn.to(device)

    optimizer = optim.Adam(dcn.parameters(), lr=0.0001)
    lossF = nn.CrossEntropyLoss()

    train_model(dcn, train_loader=train_loader, epochs=5, optimizer=optimizer, lossF=lossF, device=device,
                writer=writer,test_loader=test_loader,record_val_test=True)
    _, _ = test_model(dcn, test_loader=test_loader, device=device, lossF=lossF)

    # torch.save(dcn.state_dict(), './model_weights/dcn_model1')


if __name__ == "__main__":
    main()
