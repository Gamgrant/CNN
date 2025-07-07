import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        """
        My custom ResidualBlock

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size

        [hint]
        * See the instruction PDF for details
        * Set the bias argument to False
        """

        ## Define all the layers
        # ----- TODO -----

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        # raise NotImplementedError

        # if the main longer path is choosen
        self.convolut1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=True)
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.convolut2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=True)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)

        # if the shorter path is chosen
        self.shortConv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=True)
        self.shortBatchNorm =nn.BatchNorm2d(out_channels)


    def forward(self, x):

        # ----- TODO -----
        # raise NotImplementedError
        short =self.shortConv(x)
        short_con = self.shortBatchNorm(short)

        main_con = self.convolut1(x)
        main_con = self.batchNorm1(main_con)
        main_con= self.relu(main_con)
        main_con = self.convolut2(main_con)
        main_con = self.batchNorm2(main_con)

        # combine main con adn short con
        combined =  main_con + short_con
        output = self.relu(combined)
        return output

    
class MyResnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        """
        My custom ResNet.

        [input]
        * in_channels  : input channel number
        * num_classes  : number of classes

        [hint]
        * See the instruction PDF for details
        * Set the bias argument to False
        """

        ## Define all the layers
        # ----- TODO -----
        # raise NotImplementedError
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels=64,kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # res blocks includes main and shortcut paths
        self.res_block_1 = ResidualBlock(64, 128, kernel_size=3, stride=2)
        self.res_block_2 = ResidualBlock(128, 256, kernel_size=3, stride=2)
        self.res_block_3 = ResidualBlock(256, 512, kernel_size=3, stride=2)
        # Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # fc
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_embed=False):
        """
        Forward path.

        [input]
        * x             : input data
        * return_embed  : whether return the feature map of the last conv layer or not

        [output]
        * output        : output data
        * embedding     : the feature map after the last conv layer (optional)

        [hint]
        * See the instruction PDF for network details
        * You want to set return_embed to True if you are dealing with CAM
        """

        # ----- TODO -----
        # raise NotImplementedError
        x = self.block(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)


        if return_embed:
          embedding = x
          x = self.avgpool(x)
          x = torch.flatten(x, 1)
          out = self.fc(x)
          return out, embedding
        else:
          x = self.avgpool(x)
          x = torch.flatten(x, 1)
          out = self.fc(x)
          return out
def init_weights_kaiming(m):

    """
    Kaming initialization.

    [input]
    * m : torch.nn.Module

    [hint]
    * Refer to the course slides/recitations for more details
    * Initialize the bias term in linear layer by a small constant, e.g., 0.01
    """

    if isinstance(m, nn.Conv2d):
        # ----- TODO -----
        # raise NotImplementedError
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


    elif isinstance(m, nn.Linear):
        # ----- TODO -----
        # raise NotImplementedError
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0.01)

if __name__ == "__main__":

    # set model
    net = MyResnet(in_channels=3, num_classes=10)
    net.apply(init_weights_kaiming)

    # sanity check
    input = torch.randn((64, 3, 32, 32), requires_grad=True)

    output = net(input)
        # output = summary(net,input)

    print(output.shape)
    print("Expected output size:torch.Size([64, 10])" )   
    # expected sizes:
    # input
    # (64, 3, 32, 32)
    # first block:
    # (64, 64, 32, 32)
    # after first residual block:
    # (64, 128, 16, 16)
    # after second residual block:
    # (64, 256, 8, 8)
    # after third residual block
    # (64, 512, 4, 4)
    # after avg pooling:
    # (64, 512, 1, 1)
    # after flattening
    # (64, 512)
    # after fc
    # (64, 10)
