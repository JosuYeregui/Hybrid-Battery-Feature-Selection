import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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


df = pd.read_csv("Data/Clean_Data.csv")
data = df.drop(columns=["split", "test"])
print(df.head())
print(data.shape[0])
print(df.columns)

(X_train, y_train, y_train_sim), (X_val, y_val, y_val_sim), (X_test, y_test, y_test_sim) = \
    [(x.drop(columns=["E_real", "Ecell"]), x["E_real"], x["Ecell"]) for _, x in data.groupby(df['split'])]
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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10000):
    # Forward pass
    y_pred = model(torch.from_numpy(X_train_norm).float())

    # Compute and print loss
    loss = criterion(y_pred, torch.from_numpy(y_train.values.reshape(-1, 1)).float())
    print(epoch, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_sim = y_test_sim.values.reshape(-1, 1)
y_real = y_test.values.reshape(-1, 1)
y_model = model(torch.from_numpy(X_test_norm).float()).detach().numpy()
loss = criterion(torch.from_numpy(y_sim).float(), torch.from_numpy(y_real).float())
print(loss.item())

plt.plot(y_sim, label="Simulated")
plt.plot(y_model, label="Model")
plt.plot(y_real, label="Real")
plt.legend()
plt.show()
# X_train.hist()
# plt.show()
