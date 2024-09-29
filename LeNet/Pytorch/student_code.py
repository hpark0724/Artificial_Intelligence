# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        # convolution layer with the number of output channels to be 6, kernel size to be 5, stride to be 1
        self.conv1 =  nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, bias = True)
        # 2D max pooling layer (kernel size to be 2 and stride to be 2)
        self.max_pool_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # convolution layer with the number of output channels to be 16, kernel size to be 5, stride to be 1
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels =16, kernel_size = 5, stride = 1, bias = True)
        # 2D max pooling layer (kernel size to be 2 and stride to be 2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # convert matrix with 5 * 5 * 16 (=400) features to a linear layer with output dimension to be 256
        self.linear_layer1 = nn.Linear(16*5*5, 256, bias = True)
        # a linear layer with output dimension to be 128
        self.linear_layer2 = nn.Linear(256, 128, bias = True)
        # a linear layer with output dimension to be 100
        self.linear_layer3 = nn.Linear(128, 100, bias = True)


    def forward(self, x):
        shape_dict = {}
        # certain operations
        # convolution layer followed by a relu activation layer
        x = nn.functional.relu(self.conv1(x))
        # 2D max pooling layer (kernel size to be 2 and stride to be 2)
        x = self.max_pool_1(x)
        # store the output in the list after max pooling layer
        shape_dict[1] = list(x.size())
        # convolution layer followed by a relu activation layer
        x = nn.functional.relu(self.conv2(x))
        # 2D max pooling layer (kernel size to be 2 and stride to be 2)
        x = self.max_pool_2(x)
        # store the output in the list after second max pooling layer
        shape_dict[2] = list(x.size())

        # flatten max_pool_2 to contain 16*5*5 columns
        x = x.view(-1, 400)
        # store the output in the list after flatten
        shape_dict[3] = list(x.size())
        # perform relu activation function
        x = nn.functional.relu(self.linear_layer1(x))
        # store the output in the list after relu activation function
        shape_dict[4] = list(x.size())
        # perform relu activation function
        x = nn.functional.relu(self.linear_layer2(x))
        # store the output in the list after relu activation function
        shape_dict[5] = list(x.size())
        # perform a linear layer activation function
        x = self.linear_layer3(x)
        # store the output in the list after relu activation function
        shape_dict[6] = list(x.size())
        # the final output
        out = x
 
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    # create the LeNet model
    model = LeNet()
    model_params = 0.0
    # interate over the parameters in the model
    for param in model.named_parameters():
        # add all the value of the product of all elements in the tensor
        model_params += torch.prod(torch.tensor(param[1].size()))    # Divide the number of trainable parameters by 1e6 (in millions)
    model_params = model_params/ 1e6
    # return the number of trainable parameters (in millions)
    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
