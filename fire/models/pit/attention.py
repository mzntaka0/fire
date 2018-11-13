# -*- coding: utf-8 -*-
"""
Attention Classes.

Several mode of attention exists here.
You can create new attention class naming "*Attention", where you can name with any string at *.  After adding new attention class, you have to add mode to __modes class variable in Attention class(first written class below).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    """
    This Attention is contexted by the paper, 'Attention is All You Need'

    Reference:
    http://deeplearning.hatenablog.com/entry/transformer

    Args:
    """
    __modes = [
            'general',
            'concat'
            ]

    def __init__(self, input_size, hidden_size, window_size, attention_mode='general'):
        super().__init__()
        if not attention_mode in self.__modes:
            raise ValueError('Attention mode must be from {}'.format(
                self.modes
                ))
        self.attention_mode = attention_mode
        self.attentions = self._get_attention_classes_from_global()
        self.score = self.init_attention(attention_mode, input_size, hidden_size)
        self.h = None
        self.N = window_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, s_before, h_sliced):
        if self.h is None:
            raise ValueError('Please set h of instance variable.')
        denom = self.score_denom(s_before)
        #print('denom', denom.size())
        scores = self.score(s_before, h_sliced) / denom
        scores = scores.transpose(0, 2)
        #print('attn_scores', scores.size())
        #print('attn_h_sliced', h_sliced.size())
        c_t = (scores*h_sliced).sum(0, keepdim=True)
        #print('attn c_t', c_t.size())
        return c_t.squeeze(0)

    def _get_attention_classes_from_global(self):
        globals_ = globals()
        attentions = {key.replace('Attention', '').lower(): value for key, value in globals().items() if 'Attention' in key}
        return attentions

    def init_attention(self, attention_mode, *args, **kwargs):
        return self.attentions[attention_mode](*args, **kwargs)

    def score_denom(self, s_before):
        #denom = torch.tensor([self.score(s_before, h_i) for h_i in self.h]).sum()
        denom = self.score(s_before, self.h).sum(2, keepdim=True)
        return denom

    def _device(self, device):
        self.device = device

class GeneralAttention(nn.Module):

    def __init__(self, input_size, s_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1)
        #self.W_a = nn.Parameter(torch.randn(input_size, output_size))
        self.h = None

    def forward(self, s_before, h_k):
        middle_output = self.fc1(h_k)
        output = (s_before*middle_output).sum(2, keepdim=True)
        output = output.transpose(0, 2)
        #print('scoreoutput', output.size())
        return output


# TODO: have to decide input_size, output_size from s, h. v_a size also should be decided.
class ConcatAttention(nn.Module):
    """
    Output score

    score(s_before, h_k) = v_a.T * tanh(W_a*cat([s_before, h_k]))
    """

    def __init__(self, input_size, output_size, memory_size):
        super().__init__()
        #self.W_a = nn.Parameter(torch.randn(input_size, output_size))
        #self.v_a = nn.Parameter(torch.randn(output_size))
        self.fc1 = nn.Linear(input_size, memory_size)
        self.fc2 = nn.Linear(memory_size, output_size)

    # FIXME: invalid size
    def forward(self, s_before, h_k):
        x = torch.cat(s_before, h_k)
        medium_output = F.tanh(self.fc1(x))
        output = self.fc2(medium_output)
        #medium_output = torch.tanh(torch.mv(self.W_a, x))
        #output = torch.dot(self.v_a, medium_output)
        return output


if __name__ == '__main__':
    attention = Attention(10, 10, 10, 'general')
    attention.score.h = 'hoge'
    #print(attention.score.h)
