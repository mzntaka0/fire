# -*- coding: utf-8 -*-
"""
"""
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Decoder(nn.Module):
    """

    Args:
    """
    rnn_modules = [
            'LSTM',
            'GRU'
            ]

    def __init__(self, input_size, output_size, hidden_size, window_size, rnn_layers, attention_modulelist, num_speaker=2, module='LSTM'):
        super().__init__()
        self._module_validation(module)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_speaker = num_speaker
        self.rnn_layers = rnn_layers
        self.module = module
        self.attention_modulelist = attention_modulelist
        self.rnncells = RNNCells(input_size, hidden_size, rnn_layers)
        self.fc = nn.Linear(hidden_size+input_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, h):
        h_seqlen, batch_size, feature_len = h.size()  # h sequence length
        # padding to out of h
        pad = torch.zeros(self.window_size, batch_size, feature_len, device=self.device)
        h_pad = torch.cat([torch.cat([pad, h]), pad])
        #print('h_pad', h_pad.size())
        s_before = torch.randn(h.size()[1], self.hidden_size, device=self.device)  # TODO: how to set
        # set h to attention
        for attention in self.attention_modulelist:
            attention.h = h

        #output_dict = OrderedDict({i: list() for i in range(self.num_speaker)})
        output_dict = OrderedDict({i: torch.zeros(h_seqlen, batch_size, self.output_size, device='cuda') for i in range(self.num_speaker)})
        for speaker_id, attention in enumerate(self.attention_modulelist):
            #print('speaker{}'.format(speaker_id))
            for i in range(h_seqlen):
                # init inputs
                h0 = torch.randn(self.rnn_layers, batch_size, self.hidden_size, device=self.device)
                c0 = torch.randn(self.rnn_layers, batch_size, self.hidden_size, device=self.device)
                hidden_hook = [(h, c) for h, c in zip(h0, c0)]
                h_sliced = h[i:i + 2*self.window_size + 1]
                c_t = attention(s_before, h_sliced)
                skip_c_t = c_t.clone()
                s_t, hidden_hook = self.rnncells(c_t, hidden_hook)
                s_before = s_t.clone()
                concated = torch.cat([s_t, skip_c_t], dim=1)
                output = F.relu(self.fc(concated)) 
                output_dict[speaker_id][i].copy_(output)
        return list(output_dict.values())

            
    def _rnn_modulelist(self, input_size, hidden_size, rnn_layers):
        make_cell = lambda input_size, hidden_size: getattr(nn, '{}Cell'.format(self.module))(input_size=input_size, hidden_size=hidden_size) 
        return nn.ModuleList([make_cell(input_size, hidden_size)
                for i in range(rnn_layers)])

    def _module_validation(self, module):
        if not module in self.rnn_modules:
            #raise ModuleNotFoundError(
            raise ValueError(
                    '"{}" module of RNN is not available. Use from {}'.format(
                        module, self.rnn_modules
                        ))

    def _device(self, device):
        self.device = device


class RNNCells(nn.Module):
    rnn_modules = [
            'LSTM',
            'GRU'
            ]

    def __init__(self, input_size, hidden_size, rnn_layers, module='LSTM'):
        super().__init__()
        self.module = module
        self._module_validation(module)
        self.rnncells = self._rnn_modulelist(input_size, hidden_size, rnn_layers)

    def forward(self, c_t, hiddens):
        hidden_hook = list()
        for rnncell, hidden in zip(self.rnncells, hiddens):
            #print('rnncell_c_t', c_t.size())
            c_t, h_t = rnncell(c_t, hidden)
            #print('rnncell_output_c_t', c_t.size())
            hidden_hook.append(h_t)
        else:
            s_t = c_t.clone()
        return s_t, hidden_hook

    def _rnn_modulelist(self, input_size, hidden_size, rnn_layers):
        make_cell = lambda input_size, hidden_size: getattr(nn, '{}Cell'.format(self.module))(input_size=input_size, hidden_size=hidden_size) 
        module_list = nn.ModuleList([make_cell(input_size, hidden_size)])
        return module_list.extend([make_cell(hidden_size, hidden_size)
                for i in range(rnn_layers)])


    def _module_validation(self, module):
        if not module in self.rnn_modules:
            #raise ModuleNotFoundError(
            raise ValueError(
                    '"{}" module of RNN is not available. Use from {}'.format(
                        module, self.rnn_modules
                        ))




if __name__ == '__main__':
    from attention import Attention
    import torch.nn as nn
    input_size = 10
    output_size = 26 
    hidden_size = 20
    window_size = 5
    rnn_layers = 4
    batch_size = 30

    attentions = nn.ModuleList([Attention(input_size, hidden_size, window_size, attention_mode='general') for i in range(2)])
    decoder = Decoder(input_size, output_size, hidden_size, window_size, rnn_layers, attentions, module='LSTM')
    a = torch.randn(300, batch_size, input_size)
    x = a[:, 0, :]
    plt.imshow(x.numpy())
    plt.show()
    #print(decoder)
    #print('input_size', input_size)
    #print('output_size', output_size)
    #print('hidden_size', hidden_size)
    #print('window_size', window_size)
    #print('rnn_layers', rnn_layers)

    output = decoder(a)
    #print(output)
    output0 = 256*output[0][0]
    plt.imshow(output0.type(torch.uint8))
    plt.show()
