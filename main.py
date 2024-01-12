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

  def __init__(self, in_features = 10, h1 = 15, h2 = 16, h3 = 12, h4 = 10, out_features = 46, dropout_rate=0.2):
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

X = data_df.drop([ 'Rank', 'Position', 'Year', 'Name', 'Age', 'Plate_Appearances', 'Doubles', 'Triples', 'Caught_Stealing', 'Base_On_Balls', 'On_Base_Percentage', 'Slugging_Percentage', 'On_Base_Plus_Slugging_Percentage_Plus', 'Total_Bases', 'Double_Plays_Grounded_Into', 'Times_Hit_By_Pitch', 'Sacrifice_Hits', 'Sacrifice_Flies', 'Intentional_Bases_on_Balls', 'Dominant_Hand', 'Switch_Hitter' ], axis=1)
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

epochs = 7001 # Going to start with a large number of Epochs to ensure data is well trained
losses = []
print('\nTraining model, this may take a few minutes...')

for i in range(epochs):
  y_pred = model.forward(X_train)

  loss = criterion(y_pred, y_train) # Test predicted values against training values

  losses.append(loss.detach().numpy())

  if i % 1000 == 0:
    print(f'\tEpoch {i}/{epochs-1} had a loss of {loss}.')

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


# Graph loss over time
plt.plot(range(epochs), losses)
plt.title('Loss over Training Period')
plt.ylabel('loss/error')
plt.xlabel('Epoch')
plt.show()

# Test model
with torch.no_grad():
  y_eval = model.forward(X_test)
  loss = criterion(y_eval, y_test)
  # print(loss) # Test loss

correct = 0
within = [0, 0, 0, 0] # Will store the number of correct within 2, 3, 4, and 5 years respectfully

with torch.no_grad():
  for i, data in enumerate(X_test):
    y_val = model.forward(data)

    if y_val.argmax().item() == y_test[i]:
      correct += 1
    if (y_val.argmax().item() > y_test[i] and y_val.argmax().item() < y_test[i] + 2) or (y_val.argmax().item() < y_test[i] and y_val.argmax().item() > y_test[i] - 2):
      within[0] += 1
    if (y_val.argmax().item() > y_test[i] and y_val.argmax().item() < y_test[i] + 3) or (y_val.argmax().item() < y_test[i] and y_val.argmax().item() > y_test[i] - 3):
      within[1] += 1
    if (y_val.argmax().item() > y_test[i] and y_val.argmax().item() < y_test[i] + 4) or (y_val.argmax().item() < y_test[i] and y_val.argmax().item() > y_test[i] - 4):
      within[2] += 1
    if (y_val.argmax().item() > y_test[i] and y_val.argmax().item() < y_test[i] + 5) or (y_val.argmax().item() < y_test[i] and y_val.argmax().item() > y_test[i] - 5):
      within[3] += 1


# Output results of test
print(f'\nThe model predicted {correct} right out of {len(y_test)}.')
print(f'\tand it predicted {within[0]} within 2 years out of {len(y_test)}.')
print(f'\tand it predicted {within[1]} within 3 years out of {len(y_test)}.')
print(f'\tand it predicted {within[2]} within 4 years out of {len(y_test)}.')
print(f'\tand it predicted {within[3]} within 5 years out of {len(y_test)}.')

torch.save(model.state_dict(), 'ai_mlb_age_predictor.pt') # Save model

if input('\nWould you like to enter player data for yourself? [y/n]\n\t> ') == 'y':
  while True:
    print('\n')

    games = float(input('Enter the amount of games the player played in the season you want to analyze: '))
    pa = float(input('Enter the amount of at bats: '))
    runs = float(input('Enter the amount of runs: '))
    hits = float(input('Enter the amount of hits: '))
    hr = float(input('Enter the amount of home runs: '))
    rbi = float(input('Enter the amount of RBI: '))
    sb = float(input('Enter the amount of stolen bases: '))
    strikeouts = float(input('Enter the amount of strikeouts: '))
    ba = float(input('Enter the batting average (eg: .394): ' ))
    ops = float(input('Enter the players OPS (eg: 0.932): '))

    player_data = torch.tensor([games, pa, runs, hits, hr, rbi, sb, strikeouts, ba, ops])

    print('I predict this players age is...')

    with torch.no_grad():
      y_val = model.forward(player_data)
      print(y_val.argmax().item())
    
    if input('\nWould you like to try another player? [y/n]\n\t> ') != 'y':
      break
