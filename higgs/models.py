import torch

from torch import nn


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(21, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(0.25),
      nn.Linear(128, 1)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)