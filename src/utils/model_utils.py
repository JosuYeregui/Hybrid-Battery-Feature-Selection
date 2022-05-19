import torch
from torch import optim
import time
import matplotlib.pyplot as plt
from .models import *


def train_model(model, dataloaders, criterion, optimizer, num_epochs=1000, device='cpu'):
    """
    Trains a model for a given number of epochs.
    :param model: PyTorch model to train.
    :param dataloaders: PyTorch dataloaders for training and validation.
    :param criterion: PyTorch loss function.
    :param optimizer: PyTorch optimizer.
    :param num_epochs: Number of epochs to train for.
    :param device: Device to train on.
    :return: Trained model.
    """
    since = time.time()

    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        if (epoch + 1) % 100 == 0:
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
                labels = labels.to(device)

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
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            history[phase + '_loss'].append(epoch_loss)
            if (epoch + 1) % 100 == 0:
                print('\t{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss

        if (epoch + 1) % 100 == 0:
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss), end='\n\n')

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

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)

    print('\tLoss: {:.4f}'.format(epoch_loss))

    return epoch_loss


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


def MAPE(y_pred, y_true):
    """
    Calculates the mean absolute percentage error.
    :param y_true: True values.
    :param y_pred: Predicted values.
    :return: Mean absolute percentage error.
    """
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100


def initialize_model(model_name, input_features, device='cpu', lr=0.001, weight_decay=0.0001):
    """
    Initializes a model.
    :param model_name: Name of the model.
    :param input_features: Number of input features.
    :param lr: Learning rate.
    :param weight_decay: Weight decay.
    :return: Model.
    """
    if model_name == 'fnn':
        model = FFNN(input_features, 64, 1).to(device)
    elif model_name == 'cnn':
        model = CNN_1D(input_features, 128).to(device)
    elif model_name == 'lstm':
        model = LSTM(input_features, 64, 1, device).to(device)
    else:
        raise ValueError('Model not recognized.')

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
