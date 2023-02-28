import numpy as np
from torch import optim
import time
import matplotlib.pyplot as plt
from .models import *
import copy
import scienceplots
import mpl_toolkits
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset


def train_model(model, dataloaders, criterion, optimizer, num_epochs=1000, device='cpu', early_stopping=False,
                patience=10, compute_time_per_epoch=False):
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

    t_per_epoch = []
    for epoch in range(num_epochs):
        start_time = time.time()
        if (epoch + 1) % 1 == 0:
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
            if (epoch + 1) % 1 == 0:
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

        t_per_epoch.append(time.time() - start_time)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss), end='\n\n')

    if early_stopping:
        model = early.best_model

    if compute_time_per_epoch:
        return model, history, t_per_epoch
    return model, history


def model_evaluation(model, dataloader, criterion, device='cpu', return_exec_time=False):
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
    running_exec = 0.0
    y_pred = []

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            start_time = time.time()
            outputs = model(inputs)
            running_exec += time.time() - start_time
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
    exec_time = running_exec / dataloader.dataset.X.size(0) * 1e3

    print('\tLoss: {:.4f}\tAverage execution time: {:.4f}ms'.format(epoch_loss, exec_time), end='\n\n')

    if return_exec_time:
        return y_pred, epoch_loss, exec_time
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
        model = FFNN(input_features, 256, 1).to(device)
    elif model_name == 'cnn':
        model = CNN_1D(input_features, 256).to(device)
    elif model_name == 'lstm':
        model = LSTM(input_features, 256, 1, device).to(device)
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


def RMSE(y_pred, y_true):
    """
    Calculates the root mean squared error.
    :param y_true: True values.
    :param y_pred: Predicted values.
    :return: Root mean squared error.
    """
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


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


def plot_curves(t, y_real, y_sim, y_pred, y_FOM, model_name, x_label="Time [h]", y_label="Voltage [V]",
                save=False, filename=""):
    """
    Plot the prediction against the real data and simulation data
    """
    with plt.style.context(['science', 'grid', 'high-contrast']):
        cm = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(figsize=(2.333, 1.75), dpi=600.)
        ax.plot(t, y_sim, label="ROM Model", color=cm[0], lw=0.5)
        ax.plot(t, y_FOM, label="FOM Model", color="springgreen", lw=0.5)
        ax.plot(t, y_pred, label="Seq. " + model_name + " Model", color="#E9002D", lw=0.5)
        ax.plot(t, y_real, label="Experimental", color="black", lw=1)
        handles, labels = ax.get_legend_handles_labels()
        order = [3, 0, 1, 2]
        # sort both labels and handles by labels
        ax.legend([handles[i] for i in order], [labels[i] for i in order])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.autoscale(tight=True)
        ax.set_ylim((0., 100.))

        axins = zoomed_inset_axes(ax, 4, loc='lower left')
        axins.plot(t, y_sim, label="ROM Model", color=cm[0], lw=0.5)
        axins.plot(t, y_FOM, label="FOM Model", color="springgreen", lw=0.5)
        axins.plot(t, y_pred, label="Seq. " + model_name + " Model", color="#E9002D", lw=0.5)
        axins.plot(t, y_real, label="Experimental", color="black", lw=1)

        x1, x2, y1, y2 = 5, 6, 50, 60
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        axins.set_xticklabels([])
        axins.set_yticklabels([])

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        if save:
            fig.savefig(filename + ".pdf")
        plt.show()


def plot_cdf(plot_dict, xlabel="Absolute Error [%]", ylabel="CDF", scaling=100, save=False, filename=""):
    """
    Plot the CDF of the predictions.
    :param plot_dict: Dictionary containing the CDF of the predictions.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param scaling: Scaling factor.
    :param save: Save figure
    :param filename: Figure filename
    """
    with plt.style.context(['science', 'grid']):
        fig, ax = plt.subplots(dpi=600.)
        for key, value in plot_dict.items():
            ax.plot(value["pdf"]*scaling, value["cdf"], label=key, color=value["color"], ls=value["linestyle"], lw=1.5)
        ax.legend(fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 0.1*scaling)
        ax.set_ylim(0, 1)
        if save:
            fig.savefig(filename+".pdf")
        plt.show()


def plot_time_models(plot_dict, xlabel="Time per step [ms]", ylabel="Absolute SOC error [\%]", save=False, filename=""):
    """
    Plot the CDF of the predictions.
    :param plot_dict: Dictionary containing the CDF of the predictions.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param scaling: Scaling factor.
    :param save: Save figure
    :param filename: Figure filename
    """
    dt_sim_model = plot_dict["LSTM"][0]
    dt_sim_ROM = plot_dict["ROM"][0]
    dt_sim_ML = plot_dict["RawML"][0]
    dt_sim_FOM = plot_dict["FOM"][0]

    y_pred_LSTM = plot_dict["LSTM"][1]
    y_pred_LSTM_k0 = plot_dict["RawML"][1]
    y_ROM = plot_dict["ROM"][1]
    y_FOM = plot_dict["FOM"][1]
    y_real = plot_dict["Real"]

    with plt.style.context(['science', 'grid', 'high-contrast']):
        cm = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(dpi=600.)
        positions = [np.mean(dt_sim_ROM) * 1e3, np.mean(dt_sim_FOM) * 1e3,
                                      np.mean(dt_sim_ML) * 1e3, np.mean(dt_sim_model) * 1e3]

        w = 0.1
        width = lambda p, w: 10 ** (np.log10(p) + w / 2.) - 10 ** (np.log10(p) - w / 2.)
        bplot = ax.boxplot([np.abs(y_real - y_ROM), np.abs(y_real - y_FOM), np.abs(y_real - y_pred_LSTM_k0), np.abs(y_real - y_pred_LSTM)],
                           positions=positions, autorange=True,
                           manage_ticks=False, showfliers=False, widths=width(positions,w), patch_artist=True)
        colors = [cm[0], 'springgreen', '#FEBE00', '#E9002D']
        for patch, median, color in zip(bplot['boxes'], bplot['medians'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#3A3B3C")
            patch.set_lw(1.)
            patch.set_alpha(0.5)
            median.set_color('#3A3B3C')
            median.set_lw(1.)

        ax.scatter(dt_sim_ROM[::20] * 1e3, np.abs(y_real[::20] - y_ROM[::20]), alpha=0.1, s=0.1, facecolors='none', edgecolors='orange')
        ax.scatter(dt_sim_FOM[::20] * 1e3, np.abs(y_real[::20] - y_FOM[::20]), alpha=0.1, s=0.1, facecolors='none', edgecolors='orange')
        ax.scatter(dt_sim_model[::20] * 1e3, np.abs(y_real[::20] - y_pred_LSTM[::20]), alpha=0.1, s=0.1, facecolors='none',
                   edgecolors='orange')
        ax.scatter(dt_sim_ML * np.ones_like(y_pred_LSTM_k0[::20]) * 1e3, np.abs(y_real[::20] - y_pred_LSTM_k0[::20]),
                   alpha=0.1, s=0.1, facecolors='none', edgecolors='orange')

        ax.legend(bplot["boxes"], ["ROM Model", "FOM Model", "LSTM Model", "Seq. LSTM Model"], loc="upper right", fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.set_ylim([0, 3.2])
        ax.grid('both')

        if save:
            fig.savefig(filename+".pdf")

        plt.show()
