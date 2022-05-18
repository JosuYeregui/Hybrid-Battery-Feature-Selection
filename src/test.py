import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

class CNN_1D(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, 5, padding="same")
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 5, padding="same")
        self.conv3 = nn.Conv1d(hidden_size, 1, 5, padding="same")

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x.view(-1, 1)


df = pd.read_csv("Data/Clean_Data_Full.csv")
data = df.drop(columns=["split", "test"])
print(df.head())
print(data.shape[0])
print(df.columns)

(X_train, y_train, y_train_sim), (X_val, y_val, y_val_sim), (X_test, y_test, y_test_sim) = \
    [(x.drop(columns=["E_real", "Ecell", "time"]), x["E_real"], x["Ecell"]) for _, x in data.groupby(df['split'])]

X_train = df[df["test"] != 302]
y_train = X_train["E_real"]
y_train_sim = X_train["Ecell"]
X_train = X_train.drop(columns=["E_real", "Ecell", "time", "split", "test"])

X_test = df[df["test"] == 302]
y_test = X_test["E_real"]
y_test_sim = X_test["Ecell"]
X_test = X_test.drop(columns=["E_real", "Ecell", "time", "split", "test"])
print(X_train.head())
print(X_val.head())
print(X_test.head())

proc_X = preprocessing.StandardScaler().fit(X_train)
k = X_train.shape[1]
k_best = SelectKBest(mutual_info_regression, k=k).fit(proc_X.transform(X_train), y_train) # 30/5
print(k_best.get_feature_names_out(X_train.columns))
print(k_best.scores_)

X_train_norm = proc_X.transform(X_train)
X_train_norm = k_best.transform(X_train_norm)
X_val_norm = proc_X.transform(X_val)
X_val_norm = k_best.transform(X_val_norm)
X_test_norm = proc_X.transform(X_test)
X_test_norm = k_best.transform(X_test_norm)

# print(np.max(X_train_norm, axis=0), np.min(X_train_norm, axis=0))

model = Net(X_train_norm.shape[1], 64, 1)
#model = LSTM(X_train_norm.shape[1], 64, 1)
#model = CNN_1D(X_train_norm.shape[1], 128)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    # Forward pass
    #y_pred = model(torch.from_numpy(X_train_norm.reshape(X_train_norm.shape[0], 1, X_train_norm.shape[1])).float())
    y_pred = model(torch.from_numpy(X_train_norm).float())

    # Compute and print loss
    loss = criterion(y_pred, torch.from_numpy(y_train.values.reshape(-1, 1)).float())# - torch.from_numpy(y_train_sim.values.reshape(-1, 1)).float())
    if epoch % 100 == 0:
        print(epoch, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_sim = y_test_sim.values.reshape(-1, 1)
y_real = y_test.values.reshape(-1, 1)
y_model = model(torch.from_numpy(X_test_norm).float()) #+ torch.from_numpy(y_test_sim.values.reshape(-1, 1)).float()
#y_model = model(torch.from_numpy(X_test_norm.reshape(X_test_norm.shape[0], 1, X_test_norm.shape[1])).float())# + torch.from_numpy(y_train_sim.values.reshape(-1, 1)).float()
loss_1 = criterion(torch.from_numpy(y_sim).float(), torch.from_numpy(y_real).float())
loss_2 = criterion(y_model, torch.from_numpy(y_real).float())
y_model = y_model.detach().numpy()
print(loss_1.item())
print(loss_2.item())

plt.plot(y_real, label="Real", color="black")
plt.plot(y_sim, label="P2D Model", color="grey")
plt.plot(y_model, label="ML Model", color="red")
plt.title("Feature selection")
plt.legend()
plt.grid("on")
plt.show()

# Sample change
# X_train.hist()
# plt.show()
