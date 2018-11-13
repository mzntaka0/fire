# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import math
from bpdb import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F

#from ..exception import ModuleNotFoundError


class Encoder(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size=768, gcnn_num=2, rnn_layers=4, rnn_module='LSTM'):
        super().__init__()
        self.gcnn_num = gcnn_num
        self.rnn_layers = rnn_layers
        self.gcnn = nn.Sequential()
        self._num_GatedCNN(gcnn_num)
        self.birnn = BiRNN(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_layers, output_size=output_size, module=rnn_module)
        self.device = None

    def forward(self, x):
        x = self.gcnn(x)
        x = self._img2seq(x)
        x = self.birnn(x)
        #x = F.softmax(x)
        return x

    def _num_GatedCNN(self, gcnn_num):
        for i in range(gcnn_num):
            self.gcnn.add_module("GatedCNN{}".format(i + 1), GatedCNN())

    def _device(self, device):
        self.device = device

    def _img2seq(self, x):
        x = x.squeeze(1)
        x = x.transpose(0, 1).transpose(0, 2)
        return x

    def _seq2img(self, x):
        x = x.transpose(0, 2).transpose(0, 1)
        x = x.unsqueeze(1)
        return x

        

class GatedCNN(nn.Module):
    """
    Gated Convolutional Neural Network Layer.
    It is used as 1 layer. So if you wanna increase GatedCNN layer,
    you can create several instances of this as you want and join them.

    Output size and channel are as exactly same as input size, using padding.

    Args:
        - in_channels[int]:
        - out_channels[int]:

    Usage:
    >>> cnn = GatedCNN(*args, **kwargs)
    >>> output = cnn(x)
    """

    # TODO: add padding to make output size be same as input size.
    def __init__(self, in_channels=1, out_channels=1, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=self._simple_same_padding(kernel_size))
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=self._simple_same_padding(kernel_size))

    def forward(self, x):
        forward_x = self.conv(x)
        gated_x = torch.sigmoid(self.gate_conv(x))
        output = forward_x * gated_x
        return output

    def _simple_same_padding(self, kernel_size):
        """
        subject to  stride = 1
        refecence: https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
        """
        return math.floor(kernel_size / 2.0)

    def _same_padding(self, input_size, stride, dilation, kernel_size):
        """
        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

            .. math::
                H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                            \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

                W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                            \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

        """
        same_pad_f = lambda size: math.floor(((stride - 1)*size + dilation*(kernel_size - 1) + 1 - stride) / 2.0)
        return (same_pad_f(size) for size in input_size)


class BiRNN(nn.Module):
    """
    Bidirectional Recurrent Neural Network Layer.

    Args:
        - input_size[int]:
        - hidden_size[int]:
        - num_layers[int]:
        - output_size[int]:
        - module[str]:

    Usage:
    >>> rnn = BiRNN(*args, **kwargs)
    >>> output = rnn(x)
    """
    rnn_modules = [
            'LSTM',
            'GRU'
            ]
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, module='LSTM'):
        super().__init__()
        if not module in self.rnn_modules:
            #raise ModuleNotFoundError(
            raise ValueError(
                    '"{}" module of RNN is not available. Use from {}'.format(
                        module, self.rnn_modules
                        ))
        self.birnn = getattr(nn, module)(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                # batch_first=True,
                bidirectional=True
                )

    def forward(self, x):
        output = self.birnn(x)
        return output[0]  # return shape: (seq_len, batch, num_direction*hidden_size)

if __name__ == '__main__':
    import cv2
    img_path = '31909713-d9046856-b7ef-11e7-98fe-8a1e133c0010.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    # nSamples x nChannels x Height x Width
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).cuda().float()
    print(t.size())
    encoder = Encoder(222, 384).cuda()
    output = encoder(t)
    print(output)
