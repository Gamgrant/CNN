import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from mytorch import MyConv2D, MyMaxPool2D


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    def __init__(self):

        """
        My custom network
        [hint]
        * See the instruction PDF for details
        * Only allow to use MyConv2D and MyMaxPool2D
        * Set the bias argument to True
        """
        super().__init__()
        
        ## Define all the layers
        ## Use MyConv2D, MyMaxPool2D for the network
        # ----- TODO -----
        self.conv1 = MyConv2D(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = MyConv2D(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool = MyMaxPool2D(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(6*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        # raise NotImplementedError


    def forward(self, x):
        
        # ----- TODO -----
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(-1,6*7*7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        # raise NotImplementedError


if __name__ == "__main__":

    # set param
    setup_seed(18786)
    batch_size = 128
    num_epoch = 5
    lr = 1e-4

    ## Load dataset
    # ----- TODO -----
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset =  torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    print(f"LOAD DATASET: TRAIN {len(trainset)} | TEST: {len(valset)}")

    ## Load my neural network
    simple_net = Net()
    
    ## Define the criterion and optimizer
    # ----- TODO -----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(simple_net.parameters(),lr = lr )
    ## Training and evaluation
    ## Feel free to record the loss and accuracy numbers
    train_loss = []
    val_loss= []
    train_acc = []
    val_acc = []
    ## Hint: you could separate the training and evaluation
    ## process into 2 different functions for each epoch
    for epoch in range(num_epoch): 
        # ----- TODO -----
        simple_net.train()
        current_loss = 0
        current_correct = 0
        current_total = 0
        for idx, data in enumerate(trainloader):
            data_in, label = data
            optimizer.zero_grad()
            out = simple_net(data_in)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            max_val, max_index = torch.max(out.data,1)
            current_total += label.size(0)
            current_correct += (max_index == label).sum().item()
        train_acc.append(current_correct/current_total)
        train_loss.append(current_loss/len(trainloader))

        simple_net.eval()
        current_loss = 0 
        current_correct = 0
        current_total = 0
        with torch.no_grad():
            for data in valloader:
                data_in, label = data
                out = simple_net(data_in)
                loss = criterion(out, label)
                current_loss += loss.item()
                max_val, max_index = torch.max(out.data,1)
                current_total += label.size(0)
                current_correct +=(max_index == label).sum().item()
                #
                # out data looks like this:
                #  [[0.1, 0.2, 0.7, 0.0],  # Sample 1
                #  [0.8, 0.1, 0.0, 0.1],  # Sample 2
                #  [0.3, 0.4, 0.2, 0.1]]  # Sample 3
                # 
                #                   
                # label looks like this: [2, 0, 1]  Corresponds to the correct class for each sample
                #
                # max_index = [2,   0,   1  ]
                #(max_index == label) gives [True, True, True]
                #
        val_acc.append(current_correct/current_total)
        val_loss.append(current_loss / len(valloader))
        print(f"Epoch {epoch+1} - Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_acc[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, Val Accuracy: {val_acc[-1]:.4f}")

    ## Plot the loss and accuracy curves
    # ----- TODO -----
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Training and Validation Loss across Epochs")
    plt.savefig("loss_plot_del1-2.png")  
    plt.clf()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Training and Validation Accuracy across Epochs")
    plt.savefig("accuracy_plot_del1-2.png")  # Save the plot as a PNG file


