#%%

### STANDARD IMPORTS ###    
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

### OOP FOR TRAINING MODEL ###
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()

        # HOW MANY LAYERS?
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # PROPAGATION FUNCTION #
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


### IMPORTING CSV FILE ###
df = pd.read_csv("path to iris.csv file")

### EDITING CSV FILE ###
x = df.drop("target", axis=1)
y = df['target']
x = x.values
y = y.values

### SPLITTING DATA ###
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

### CREATING MODEL FROM MODEL CLASS ###
model = Model()

### CREATING LOSS AND OPTIMIZING FUNCTIONS ###
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

### SETTING NUMBER OF EPOCHS AND LIST OF LOSSES ###
epochs = 100
losses = []

### PERFORMING LEARNING PROCESS FOR EVERY EPOCH ###
for i in range(epochs):

    # PREDICTING Y VALUE
    y_pred = model.forward(X_train)

    # CALCULATING LOSS
    loss = criterion(y_pred, y_train)

    # APPENDING LOSS LIST WITH CURRENT LOSS
    losses.append(loss)

    # PRINTING LOSS EVERY 10TH EPOCH
    if i % 10 == 0:
        print(f'EPOCH: {i}; LOSS: {loss.item()}')

    # PERFORMING BACKPROPAGATION
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

### PREPARING OBTAINED DATA FOR GRAPHICAL OUTPUT ###
losses = torch.tensor(losses).reshape(10, -1)
epochs = [i for i in range(epochs)]
epochs = torch.tensor(epochs).reshape(10, -1)

### PRINTING GRAPHICAL VISUALISATION OF LOSSES AGAINST NUMBER OF EPOCHS ###
print(plt.plot(epochs, losses))

### TESTING MODEL ON TEST DATA ###
correct = 0

with torch.no_grad():
    y_evaluated = model.forward(X_test)
    loss = criterion(y_evaluated, y_test)

    # FINDING IF PREDICTION IS CORRECT OR NO
    for i, data in enumerate(y_evaluated):
        # print(f'{y_evaluated[i].argmax()} => {y_test[i]}')
        if y_evaluated[i].argmax() == y_test[i]:
            correct += 1
    
    # PRINTING NUMBER OF CORRECT PREDICTIONS
    print(correct)

### SAVING MODEL TO A FILE ###
torch.save(model.state_dict(), 'path where you want to save model.pt file')

### CREATING A NEW MODEL FROM A FILE ###
new_model = Model()
new_model.load_state_dict(torch.load('path to model.pt file'))

### PREDICTING A NEW FLOWER WITH A NEW MODEL
mystery_iris = torch.tensor([5.6,3.7,2.2,0.5])
with torch.no_grad():
    print(new_model(mystery_iris).argmax())
# %%
