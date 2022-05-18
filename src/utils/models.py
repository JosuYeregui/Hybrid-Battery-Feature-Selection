import torch
import torch.nn as nn
from torch.autograd import Variable


class FFNN(nn.Module):
    def __init__(self, input, H, output):
        super(FFNN, self).__init__()
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
    def __init__(self, input_size, hidden_size, kernel_size=5):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding="same")
        self.conv3 = nn.Conv1d(hidden_size, 1, kernel_size, padding="same")

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x.view(-1, 1)
