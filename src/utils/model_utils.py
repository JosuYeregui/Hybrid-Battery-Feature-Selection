import numpy as np
from torch import optim
import time
import matplotlib.pyplot as plt
from .models import *
import copy


def train_model(model, dataloaders, criterion, optimizer, num_epochs=1000, device='cpu', early_stopping=False,
                patience=10):
    """
    Trains a model for a given number of epochs.
    :param model: PyTorch model to train.
    :param dataloaders: PyTorch dataloaders for training and validation.
    :param criterion: PyTorch loss function.
    :param optimizer: PyTorch optimizer.
    :param num_epochs: Number of epochs to train for.
    :param device: Device to train on.
    :param early_stopping: Whether to use early stopping.
    :param patience: Number of epochs to wait before early stopping.
    :return: Trained model.
    """
    since = time.time()

    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    early = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        if (epoch + 1) % 10 == 0:
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.reshape(-1, 1).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                if len(inputs.size()) == 3:
                    running_loss += loss.item() * inputs.size(1)
                else:
                    running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataloaders[phase].dataset.X.size(0)

            history[phase + '_loss'].append(epoch_loss)
            if (epoch + 1) % 10 == 0:
                print('\t{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val':
                if early_stopping:
                    early(epoch_loss, model)
                    best_loss = early.best_score
                else:
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss

        if early_stopping and early.early_stop:
            print('Early stopping at epoch {}'.format(epoch))
            break

        if (epoch + 1) % 10 == 0:
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}%'.format(best_loss), end='\n\n')

    if early_stopping:
        model = early.best_model

    return model, history


def model_evaluation(model, dataloader, criterion, device='cpu'):
    """
    Evaluates a model on a given dataset.
    :param model: PyTorch model to evaluate.
    :param dataloader: PyTorch dataloaders for training and validation.
    :param criterion: PyTorch loss function.
    :param device: Device to evaluate on.
    :return: Accuracy and loss.
    """
    print('Evaluating model on test set...')
    model.eval()
    running_loss = 0.0
    y_pred = []

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if len(y_pred) == 0:
            y_pred = outputs.detach().cpu().numpy()
        else:
            y_pred = np.vstack((y_pred, outputs.detach().cpu().numpy()))

        # statistics
        if len(inputs.size()) == 3:
            running_loss += loss.item() * inputs.size(1)
        else:
            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataloader.dataset.X.size(0)

    print('\tLoss: {:.4f}%'.format(epoch_loss), end='\n\n')

    return y_pred, epoch_loss


class EarlyStopping:
    """
    Implements early stopping.
    """
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model = None

    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.best_score = score
            self.best_model = copy.deepcopy(model)
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


def initialize_model(model_name, input_features, device='cpu', lr=0.001, weight_decay=0.0001):
    """
    Initializes a model.
    :param model_name: Name of the model.
    :param input_features: Number of input features.
    :param device: Device to train on.
    :param lr: Learning rate.
    :param weight_decay: Weight decay.
    :return: Model.
    """
    if model_name == 'fnn':
        model = FFNN(input_features, 128, 1).to(device)
    elif model_name == 'cnn':
        model = CNN_1D(input_features, 256).to(device)
    elif model_name == 'lstm':
        model = LSTM(input_features, 128, 1, device).to(device)
    else:
        raise ValueError('Model not recognized.')

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer


def MAPE(y_pred, y_true):
    """
    Calculates the mean absolute percentage error.
    :param y_true: True values.
    :param y_pred: Predicted values.
    :return: Mean absolute percentage error.
    """
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100


def plot_loss(history):
    """
    Plots the training and validation loss.
    :param history: Training and validation loss.
    """
    plt.plot(history['train_loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.legend()
    plt.grid()
    plt.show()


def plot_curves(t, y_real, y_sim, y_pred, model_name):
    """
    Plot the prediction against the real data and simulation data
    """
    plt.plot(t, y_real, label="Real", color="black")
    plt.plot(t, y_sim, label="P2D Model", color="grey")
    plt.plot(t, y_pred, label=model_name + " Prediction", color="firebrick")
    plt.legend()
    plt.xlabel("Time [h]", size=12)
    plt.ylabel("Voltage [V]", size=12)
    plt.grid("on", ls=":", lw=0.5)
    plt.show()


def plot_cdf(plot_dict):
    """
    Plot the CDF of the predictions.
    :param plot_dict: Dictionary containing the CDF of the predictions.
    """
    for key, value in plot_dict.items():
        plt.plot(value["pdf"]*1000, value["cdf"], label=key, color=value["color"], ls=value["linestyle"])
    plt.legend()
    plt.xlabel("Absolute Error [mV]")
    plt.ylabel("CDF")
    plt.grid("on")
    plt.xlim(0, 250)
    plt.ylim(0, 1)
    plt.show()
