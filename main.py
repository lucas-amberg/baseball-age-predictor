import pandas as pd
import numpy as np 
import torch
import matplotlib.pyplot as plt
import torch.nn as NN
import torch.nn.functional as F 
from sklearn.model_selection import train_test_split

# Create Machine Learning Model
# We will be using 9 Inputs, 4 Hidden Layers, and 1 Output
class Model(NN.Module):

  def __init__(self, in_features = 11, h1 = 15, h2 = 16, h3 = 12, h4 = 10, out_features = 46, dropout_rate=0.2):
    super().__init__()
    self.fc1 = NN.Linear(in_features, h1)
    self.dropout = NN.Dropout(dropout_rate)
    self.fc2 = NN.Linear(h1, h2)
    self.fc3 = NN.Linear(h2, h3)
    self.fc4 = NN.Linear(h3, h4)
    self.out = NN.Linear(h4, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = F.relu(self.fc3(x))
    x = self.dropout(x)
    x = F.relu(self.fc4(x))
    x = self.out(x)

    return x
    
# Only right Willie Mays gets the seed if we're using Giants data
torch.manual_seed(24)

model = Model()

url = './data/SFG_batting.csv'

# Import Dataset
data_df = pd.read_csv(url)

# (Prints whole data set)
# print(data_df)

X = data_df.drop([ 'Position', 'Year', 'Name', 'Age', 'At_Bats', 'Doubles', 'Triples', 'Caught_Stealing', 'Base_On_Balls', 'On_Base_Percentage', 'Slugging_Percentage', 'On_Base_Plus_Slugging_Percentage_Plus', 'Total_Bases', 'Double_Plays_Grounded_Into', 'Times_Hit_By_Pitch', 'Sacrifice_Hits', 'Sacrifice_Flies', 'Intentional_Bases_on_Balls', 'Dominant_Hand', 'Switch_Hitter' ], axis=1)
y = data_df['Age']

# Prints input data
# print(X)

# Prints output data
# print(y)

# Convert to numpy
X = X.values
y = y.values

# We are going to use a test size of 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=5)

#Convert X and y to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set model to measure error
criterion = NN.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model

epochs = 20000 # Going to start with a large number of Epochs to ensure data is well trained
losses = []

for i in range(epochs):
  y_pred = model.forward(X_train)

  loss = criterion(y_pred, y_train) # Test predicted values against training values

  losses.append(loss.detach().numpy())

  if i % 1000 == 0:
    print(f'Epoch {i} had a loss of {loss}.')

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


# Graph loss over time
plt.plot(range(epochs), losses)
plt.ylabel('loss/error')
plt.xlabel('Epoch')
plt.show()

