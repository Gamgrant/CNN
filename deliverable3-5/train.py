import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import MyResnet, init_weights_kaiming
from tqdm import tqdm


def setup_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_cam(cam_img, img,name):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(cam_img, cmap='jet', alpha=0.5)
    plt.title("CAM Image")

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(cam_img, cmap='jet', alpha=0.5)  # Overlay the CAM on the image
    plt.title("CAM Blended")
    plt.savefig(f'{name}.png')
    plt.close()  # Close the figure to prevent it from displaying in the output

    plt.show()


def cam(net, inputs, labels, idx):

    """
    Calculate the CAM.

    [input]
    * net     : network
    * inputs  : input data
    * labels  : label data
    * idx     : the index of the chosen image in a minibatch, range: [0, batch_size-1]

    [output]
    * cam_img : CAM result
    * img     : raw image

    [hint]
    * Inputs and labels are in a minibatch form
    * You can choose one images from them for CAM by idx.
    """

    net.eval()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs, feature_maps = net(inputs, return_embed=True)
        feature_map = feature_maps[idx]

        ## Find the class with highest probability
        ## Obtain the weight related to that class
        # ----- TODO -----
        _, predicted_classes = torch.max(outputs, 1)
        idx_class = predicted_classes[idx].item()

        weights_class = net.fc.weight.data[idx_class]
        # print(weights_class.shape) # 512
        # print(feature_map.shape) [512, 4, 4]
        ## Calculate the CAM
        ## Hint: you can choose one of the image (idx) from the batch for the following process
        # ----- TODO -----
        num_features = feature_map.shape[0]
        weights_reshaped = weights_class.view(num_features, 1, 1)
        # calculate weighted sum of of feature maps with the target class weights
        weighted_feature_maps = weights_reshaped * feature_map # 512, 4, 4
        cam = weighted_feature_maps.sum(dim=0) # 4 x 4

        # ## Calculate the CAM
        # ## Hint: you can choose one of the image (idx) from the batch for the following process
        # # ----- TODO -----
        # cam = None
        # cam = cam.detach().cpu().numpy()


        # ## Normalize CAM
        # ## Hint: Just minmax norm and rescale every value between [0-1]
        # ## You will want to resize the CAM result for a better visualization
        # ## e.g., the size of the raw image.
        # # ----- TODO -----
        # cam_img = None
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension
        cam_img = torch.nn.functional.interpolate(cam, size=(inputs.shape[2], inputs.shape[3]), mode='bilinear', align_corners=False)
        cam_img = cam_img.squeeze().cpu().numpy()

        # ## Denormalize raw images
        # ## Hint: reverse the transform we did before
        # ## Change the image data type into uint8 for visualization
        # # ----- TODO -----
        # img = inputs[idx].permute(1,2,0).detach().cpu().numpy()
        # img = None
        img = inputs[idx].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))

        img = img * np.array(normalize_param['std']) + np.array(normalize_param['mean'])
        img = np.clip(img, 0, 1)
        return cam_img, img

if __name__ == "__main__":

    # set param
    setup_seed(18786)
    batch_size = 128
    num_epoch = 30
    lr = 1e-3
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## Set model
    ## Set the device to Cuda if needed
    ## Initialize all the parameters
    # ----- TODO -----
    net = MyResnet(in_channels=3, num_classes=10)
    net.apply(init_weights_kaiming)
    net.to(DEVICE)


    ## Create the criterion and optimizer
    # ----- TODO -----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    ## Load dataset
    normalize_param = dict(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
        )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
        transforms.Normalize(**normalize_param,inplace=True)
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(**normalize_param,inplace=True)
        ])

    # ----- TODO -----
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"LOAD DATASET: TRAIN/VAL | {len(trainset)}/{len(valset)}")

    train_loss =[]
    val_loss = []
    train_acc = []
    val_acc = []

    


    ## Training and evaluation
    ## Feel free to record the loss and accuracy numbers
    ## Hint: you could separate the training and evaluation 
    ## process into 2 different functions for each epoch
    for epoch in range(num_epoch):
    # ----- TODO -----
        net.train()
        current_loss = 0
        current_correct = 0
        current_total = 0
        for data_in, label in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epoch} - Training"):
            data_in, label  = data_in.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = net(data_in)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            _, max_idx =torch.max(output,1)
            current_total += label.size(0)
            current_correct +=(max_idx ==label).sum().item()
        train_acc.append(current_correct/current_total)
        train_loss.append(current_loss/len(trainloader))


        net.eval()
        current_loss = 0
        current_correct = 0
        current_total = 0
        with torch.no_grad():
            for data_in, label in tqdm(valloader, desc=f"Epoch {epoch+1}/{num_epoch} - Validation"):
                data_in, label  = data_in.to(DEVICE), label.to(DEVICE)
                out = net(data_in)
                loss = criterion(out, label)
                current_loss += loss.item()
                max_val, max_index = torch.max(out.data,1)
                current_total += label.size(0)
                current_correct +=(max_index == label).sum().item()
        val_acc.append(current_correct/current_total)
        val_loss.append(current_loss / len(valloader))
        print(f"Epoch {epoch+1} - Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_acc[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, Val Accuracy: {val_acc[-1]:.4f}")


    print('Finished Training')

    ## Visualization
    ## Plot the loss and acc curves
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


    cam_img_lst = []
    img_lst = []
    for i in range(100):
        dataiter = iter(valloader)
        inputs, labels = next(dataiter)
        cam_img, img = cam(net, inputs, labels, idx=i) # idx could be changed
        cam_img_lst.append(cam_img)
        img_lst.append(img)
        visualize_cam(cam_img, img,i)
    ## Plot the CAM resuls as well as raw images
    ## Hint: You will want to resize the CAM result.
    # ----- TODO -----
    

