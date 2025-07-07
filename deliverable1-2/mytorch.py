import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def MyFConv2D(input, weight, bias=None, stride=1, padding=0):
    
    """
    My custom Convolution 2D calculation.

    [input]
    * input    : (batch_size, in_channels, input_height, input_width)
    * weight   : (you have to derive the shape :-)
    * bias     : bias term
    * stride   : stride size
    * padding  : padding size

    [output]
    * output   : (batch_size, out_channels, output_height, output_width)
    """

    # assert len(input.shape) == len(weight.shape)
    assert len(input.shape) == 4
    assert weight.shape[1] == input.shape[1]

    ## padding x with padding parameter 
    ## HINT: use torch.nn.functional.pad()
    # ----- TODO -----
    x_pad = F.pad(input, (padding, padding, padding, padding), mode='constant', value=0)
    ## Derive the output size
    batch_size, input_channels, input_height, input_width = x_pad.shape
    ## Create the output tensor and initialize it with 0
    # ----- TODO -----
    output_channel, depth, h_filter, w_filter  = weight.shape

    # int((height - kernel_h + 2*padding)/stride) + 1 # since padding is already done no need to do it again
    output_height = ((input_height - h_filter) // stride) + 1
    output_width = ((input_width - w_filter) // stride) + 1
    x_conv_out    = torch.zeros((batch_size, output_channel, output_height, output_width))
    # too slow
    ## Convolution process
    # # check if the stride os off or we need additional padding:
    # if ((input_height - h_filter) % stride != 0) or ((input_width - w_filter) % stride != 0):
    #     print("Padding or stride configuration is incorrect.")
    # for batch_idx in range(batch_size):
    #     for output_chan_idx in range(output_channel):
    #         for h_out_idx in range (output_height):
    #             for w_out_idx in range(output_width):
    #                 # make sure that the depth of filter  = input_channels
    #                 if (depth!= input_channels):
    #                     print("input channels are not equal to depth of filters")
    #                 start_h = h_out_idx * stride
    #                 start_w = w_out_idx * stride
    #                 end_h = start_h + h_filter
    #                 end_w = start_w + w_filter
                    
    #                 input_window = x_pad[batch_idx,:, start_h:end_h,start_w:end_w]
    #                 filter_window = weight[output_chan_idx]
    #                 conv = torch.sum(input_window * filter_window) + (bias[output_chan_idx] if bias is not None else 0)
    #                 x_conv_out[batch_idx,output_chan_idx, h_out_idx,w_out_idx] = conv
    #                 #input window: 
                    
    ## quicker operation to transform x_pad to column
    cols = F.unfold(x_pad, kernel_size=(h_filter, w_filter), 
                    padding=0, stride=stride)
    weights_reshaped = weight.view(output_channel, -1)
    output = weights_reshaped @ cols
    if bias is not None:
        output += bias.view(-1, 1)
    x_conv_out = output.view(batch_size, output_channel, output_height, output_width)    
    return x_conv_out

class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)

        [hint]
        * The gabor filter kernel has complex number. Be careful.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_bool = bias

        ## Create the torch.nn.Paramerter for the weights and bias (if bias=True)
        ## Be careful about the size
        # ----- TODO -----
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if self.bias_bool:
            self.b = nn.Parameter(torch.rand(out_channels))
        else:
            self.b = None

        # raise NotImplementedError
            
    
    def __call__(self, x):
        
        return self.forward(x)


    def forward(self, x):
        

        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """

        # call MyFConv2D here
        # ----- TODO -----
        return MyFConv2D(x, self.W, self.b, self.stride, self.padding)

    
class MyMaxPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None):
        
        """
        My custom MaxPooling 2D layer.
        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size

        ## Take care of the stride
        ## Hint: what should be the default stride_size if it is not given? 
        ## Think about the relationship with kernel_size
        # ----- TODO -----
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride

        # raise NotImplementedError


    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """
        
        self.batch_size = x.shape[0]
        self.channel = x.shape[1]
        self.input_height = x.shape[2]
        self.input_width = x.shape[3]
        
        ## Derive the output size
        # ----- TODO -----
        self.output_height   = ((self.input_height - self.kernel_size) // self.stride) + 1
        self.output_width    = ((self.input_width - self.kernel_size) // self.stride) + 1
        self.output_channels = self.channel
        self.x_pool_out      = torch.zeros((self.batch_size, self.output_channels, self.output_height, self.output_width))
        ## Maxpooling process
        ## Feel free to use for loop

        # this is too slow:
        # for batch in range(self.batch_size):
        #     for channel in range(self.channel):
        #         for h in range(self.output_height):
        #             for w in range(self.output_width):
        #                 # Calculate the start and end indices for the current window
        #                 h_start = h * self.stride
        #                 w_start = w * self.stride
        #                 h_end = h_start + self.kernel_size
        #                 w_end = w_start + self.kernel_size

        #                 # Extract the current window and apply max pooling
        #                 window = x[batch, channel, h_start:h_end, w_start:w_end]
        #                 self.x_pool_out[batch, channel, h, w] = torch.max(window.reshape(-1))
        # return self.x_pool_out 

        # raise NotImplementedError
        cols = F.unfold(x, kernel_size=(self.kernel_size, self.kernel_size), 
                    stride=self.stride)
    
        # Max over each column to perform pooling
        cols = cols.view(self.batch_size, self.channel, self.kernel_size * self.kernel_size, -1)
        cols, _ = cols.max(dim=2)
        
        # Reshape the output
        self.x_pool_out = cols.view(self.batch_size, self.channel, self.output_height, self.output_width)
        return self.x_pool_out


if __name__ == "__main__":

    ## Test your implementation of MyFConv2D as in deliverable 1
    ## You can use the gabor filter kernel from q1.py as the weight and conduct convolution with MyFConv2D
    ## Hint: 
    ## * Be careful with the complex number
    ## * Be careful with the difference between correlation and convolution
    # ----- TODO -----
    input_tensor = torch.tensor([[
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0],
     [9.0, 10.0, 11.0, 12.0],
     [13.0, 14.0, 15.0, 16.0]]
    ]], dtype=torch.float32)

    weights = torch.tensor([[
        [[-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]]
    ]], dtype=torch.float32)

    bias = torch.tensor([0.0], dtype=torch.float32)
    stride = 1
    padding = 1
    kernel_size = 2
    pool_stride = 2
    conv_layer_custom = MyConv2D(in_channels=1, out_channels=1, kernel_size=3, stride=stride, padding=padding, bias=True)
    conv_layer_custom.W.data = weights
    conv_layer_custom.b.data = bias
    output_conv_custom = conv_layer_custom(input_tensor)
    pool_layer_custom = MyMaxPool2D(kernel_size=kernel_size, stride=pool_stride)
    output_pool_custom = pool_layer_custom(output_conv_custom)
    output_conv_builtin = F.conv2d(input_tensor, weights, bias, stride, padding)
    output_pool_builtin = F.max_pool2d(output_conv_builtin, kernel_size=kernel_size, stride=pool_stride)
    print("Custom Conv2D Output:\n", output_conv_custom)
    print("PyTorch Built-in Conv2D Output:\n", output_conv_builtin)
    print("Custom MaxPooling Output:\n", output_pool_custom)
    print("PyTorch Built-in MaxPooling Output:\n", output_pool_builtin)
    are_conv_close = torch.allclose(output_conv_custom, output_conv_builtin)
    print("Are the Conv2D outputs close?", are_conv_close)
    are_pool_close = torch.allclose(output_pool_custom, output_pool_builtin)
    print("Are the MaxPooling outputs close?", are_pool_close)


    pass
