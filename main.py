import pandas as pd
import numpy as np 
import torch
import matplotlib.pyplot as pl 
import torch.nn as NN
import torch.nn.functional as F 

# Create Machine Learning Model
# We will be using 8 Inputs, 4 Hidden Layers, and 1 Output
class Model(NN.module):

  def __init__(self, in_features = 8, h1 = 12, h2 = 12, h3 = 12, h4 = 12, out_features = 1):
    super().__init__
    self.fc1 = NN.Linear(in_features, h1)
    self.fc2 = NN.Linear(h1, h2)
    self.fc3 = NN.Linear(h2, h3)
    self.fc4 = NN.Linear(h3, h4)
    self.out = NN.Linear(h4, out_features)

    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = F.relu(self.fc4(x))
      x = F.relu(self.out(x))

      return x