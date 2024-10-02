import torch
import torch.nn as nn


@torch.no_grad()
def init_weights(m):
    """Initialize weights with Glorot uniform and biases to zero."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)  # Fixed: Weight initialization for BatchNorm
        nn.init.constant_(m.bias, 0)

class BaseNet(nn.Module):
    """Base model class common methods and attributes."""
    def __init__(self):
        super().__init__()

    def initialize(self):
        self.apply(init_weights)

