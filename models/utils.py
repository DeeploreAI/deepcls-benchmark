# Basic lib
import random

# DL lib
import torch
import torch.nn as nn
import numpy as np

# Local lib
from models.common import *


MODULE_MAP = {
    'Linear': Linear,
    'Conv': Conv,
    'ConvBN': ConvBN,
    'Bottleneck': BottleneckBlock,
    'ResNetLayer': ResNetLayer,
    'VggBlock': VggBlock,
    'AvgPool': AvgPool,
    'MaxPool': MaxPool,
    'Upsample': Upsample,
    'Concat': Concat,
    'Flatten': Flatten,
}


def parse_module(module_name, *args, **kwargs):
    if module_name not in MODULE_MAP:
        raise ValueError(f'Unknown module name: {module_name}')
    else:
        module_class = MODULE_MAP[module_name]
        module = module_class(*args, **kwargs)
        return module


class Model(nn.Module):
    """Construct a model with yaml config file."""
    def __init__(self, model_cfg):
        super().__init__()
        # Parse backbone model
        backbone_modules = nn.ModuleList()
        for i, (n_repeat, module_name, args) in enumerate(model_cfg["backbone"]):  # from, number, module, args
            for _ in range(n_repeat):
                module = parse_module(module_name, *args)
                backbone_modules.append(module)
        self.backbone_modules = nn.Sequential(*backbone_modules)  # TODO: try to use pytorch hooks to print the hidden layer results, or use torchinfo to print a table of each module

        # Parse task head
        head_modules = nn.ModuleList()
        for i, (n_repeat, module_name, args) in enumerate(model_cfg['head']):
            for _ in range(n_repeat):
                module = parse_module(module_name, *args)
                head_modules.append(module)
        self.head_modules = nn.Sequential(*head_modules)

    def forward(self, x):
        x = self.backbone_modules(x)
        x = self.head_modules(x)
        return x
