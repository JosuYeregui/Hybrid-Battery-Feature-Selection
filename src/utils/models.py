import torch
import torch.nn as nn
from torch.autograd import Variable


class FFNN(nn.Module):
    """
    Feed-forward neural network
    """
    def __init__(self, input, H, output):
        super(FFNN, self).__init__()
        self.linear1 = nn.Linear(input, H)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(H, H)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(H, output)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.linear3(x)
        return x


class LSTM(nn.Module):
    """
    LSTM model
    """
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.device = device

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        # Propagate input through LSTM
        self.lstm.flatten_parameters()
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        # Output dense layers
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out.reshape(-1, 1)


class CNN_1D(nn.Module):
    """
    1D convolutional neural network
    """
    def __init__(self, input_size, hidden_size, kernel_size=3):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding="same")
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding="same")
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding="same")
        self.conv4 = nn.Conv1d(hidden_size, 1, kernel_size, padding="same")

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        return x.view(-1, 1)
