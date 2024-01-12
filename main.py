import pandas as pd
import numpy as np 
import torch
import matplotlib.pyplot as pl 
import torch.nn as NN
import torch.nn.functional as F 

# Create Machine Learning Model
# We will be using 8 Inputs, 4 Hidden Layers, and 1 Output
class Model(NN.Module):

  def __init__(self, in_features = 8, h1 = 12, h2 = 12, h3 = 12, h4 = 12, out_features = 1):
    super().__init__()
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
    
# Only right Willie Mays gets the seed if we're using Giants data
torch.manual_seed(24)

model = Model()

url = './data/SFG_batting.csv'

# Import Dataset
data_df = pd.read_csv(url)

# (Prints whole data set)
# print(data_df)

X = data_df.drop(['Rank', 'Position', 'Name', 'Age', 'At_Bats', 'Doubles', 'Triples', 'Home_Runs', 'Stolen_Bases', 'Caught_Stealing', 'Base_On_Balls', 'On_Base_Percentage', 'Slugging_Percentage', 'On_Base_Plus_Slugging_Percentage_Plus', 'Total_Bases', 'Double_Plays_Grounded_Into', 'Times_Hit_By_Pitch', 'Sacrifice_Hits', 'Sacrifice_Flies', 'Intentional_Bases_on_Balls', 'Dominant_Hand', 'Switch_Hitter' ], axis=1)
y = data_df['Age']

# Prints input data
# print(X)

# Prints output data
# print(y)