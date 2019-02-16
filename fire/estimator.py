# -*- coding: utf-8 -*-
"""
"""
from pathlib import Path

import torch


class Estimator:

    def __init__(self, model: torch.nn.Module, trained_param_path: Path, cuda):
        self.device = torch.device('cuda' if cuda >= 0 else 'cpu')
        self.trained_param_path = trained_param_path
        self.model = model.to(self.device)
        self._load_param()
        self.model.eval()

    def run(self, x):
        y = self.model(x)
        return y

    def _load_param(self):
        param_dict = torch.load(
            self.trained_param_path,
            map_location=self.device
        )
        self.model.load_state_dict(param_dict)
