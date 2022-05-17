import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, input, H, output):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input, H)
        self.linear2 = nn.Linear(H, H)
        self.linear3 = nn.Linear(H, output)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out


df = pd.read_csv("Data/Clean_Data.csv")
data = df.drop(columns=["split", "test"])
pd.plotting.scatter_matrix(data, alpha=0.3)
plt.show()
print(df.head())
print(data.shape[0])
print(df.columns)

(X_train, y_train, y_train_sim), (X_val, y_val, y_val_sim), (X_test, y_test, y_test_sim) = \
    [(x.drop(columns=["E_real", "Ecell", "time"]), x["E_real"], x["Ecell"]) for _, x in data.groupby(df['split'])]
print(X_train.head())
print(X_val.head())
print(X_test.head())

proc_X = preprocessing.StandardScaler().fit(X_train)
k_best = SelectKBest(f_regression, k=5).fit(proc_X.transform(X_train), y_train)
print(k_best.get_feature_names_out(X_train.columns))

X_train_norm = proc_X.transform(X_train)
X_train_norm = k_best.transform(X_train_norm)
X_val_norm = proc_X.transform(X_val)
X_val_norm = k_best.transform(X_val_norm)
X_test_norm = proc_X.transform(X_test)
X_test_norm = k_best.transform(X_test_norm)

# print(np.max(X_train_norm, axis=0), np.min(X_train_norm, axis=0))

model = Net(X_train_norm.shape[1], 128, 1)
model = LSTM(X_train_norm.shape[1], 128, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    # Forward pass
    y_pred = model(torch.from_numpy(X_train_norm.reshape(X_train_norm.shape[0], 1, X_train_norm.shape[1])).float())
    #y_pred = model(torch.from_numpy(X_train_norm).float())

    # Compute and print loss
    loss = criterion(y_pred, torch.from_numpy(y_train.values.reshape(-1, 1)).float() - torch.from_numpy(y_train_sim.values.reshape(-1, 1)).float())
    print(epoch, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_sim = y_train_sim.values.reshape(-1, 1)
y_real = y_train.values.reshape(-1, 1)
#y_model = model(torch.from_numpy(X_train_norm).float()).detach().numpy() + torch.from_numpy(y_train_sim.values.reshape(-1, 1)).float().detach().numpy()
y_model = model(torch.from_numpy(X_train_norm.reshape(X_train_norm.shape[0], 1, X_train_norm.shape[1])).float()).detach().numpy() + torch.from_numpy(y_train_sim.values.reshape(-1, 1)).float().detach().numpy()
loss = criterion(torch.from_numpy(y_sim).float(), torch.from_numpy(y_real).float())
print(loss.item())

plt.plot(y_sim, label="Simulated")
plt.plot(y_model, label="Model")
plt.plot(y_real, label="Real")
plt.legend()
plt.show()


# X_train.hist()
# plt.show()
