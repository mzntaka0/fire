# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .attention import Attention


class PITASR(nn.Module):
    """

    Args:
    """

    def __init__(self, input_size, output_size, hidden_size, window_size, batch_size, gcnn_num=2, speaker_num=2, rnn_layers=4, rnn_module='LSTM'):
        super().__init__()
        self.speaker_num = speaker_num
        self.encoder = Encoder(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                gcnn_num=gcnn_num,
                rnn_layers=rnn_layers,
                rnn_module=rnn_module
                )  # TODO: enter params
        self.attention = self._attention_modulelist(2*hidden_size, output_size, window_size, speaker_num=speaker_num)
        self.decoder = Decoder(
                input_size=2*hidden_size,
                output_size=output_size,
                hidden_size=hidden_size,
                window_size=window_size,
                rnn_layers=rnn_layers,
                attention_modulelist=self.attention,
                num_speaker=speaker_num,
                module=rnn_module
                )  # decoder number = speaker number
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: deal with multi speaker
    def forward(self, x):
        h = self.encoder(x)
        outputs = self.decoder(h)
        return outputs

    def _attention_modulelist(self, input_size, hidden_size, window_size, speaker_num):
        return nn.ModuleList([Attention(input_size=input_size, hidden_size=hidden_size, window_size=window_size)
                for i in range(speaker_num)])

    def _device(self, device):
        self.device = device
        self.encoder._device(device)
        self.decoder._device(device)


if __name__ == '__main__':
    input_size = 10
    output_size = 26 
    hidden_size = 20
    window_size = 10
    rnn_layers = 4
    gcnn_num = 2
    batch_size = 30
    speaker_num = 2
    #a = torch.randn(300, batch_size, input_size)
    print('input_size', input_size)
    print('output_size', output_size)
    print('hidden_size', hidden_size)
    print('window_size', window_size)
    print('rnn_layers', rnn_layers)
    print('batch_size', batch_size)

    h = torch.randn(batch_size, 1, 300, input_size)
    pitasr = PITASR(
            input_size=input_size, 
            output_size=output_size,
            hidden_size=hidden_size,
            batch_size=batch_size,
            window_size=window_size,
            rnn_layers=rnn_layers,
            gcnn_num=gcnn_num,
            speaker_num=speaker_num
            )
    print(pitasr)
    output = pitasr(h)
    print(output[0].size())
