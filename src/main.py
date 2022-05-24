import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import pandas as pd
from utils.data_processing import *
from utils.model_utils import *


def model_training():
    """
    Main function for the application
    """
    # Set up the experiment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "lstm"
    store_path = "./results/"
    store = True
    early_stopping = True
    k = 20

    # Load the data
    print("Loading data...")
    df = pd.read_csv("./Data/Clean_Data_Full.csv")
    print("\tData loaded, shape:", df.shape, end="\n\n")

    # Process the data
    print("Processing data...", end="\n\n")
    (X_train, y_train, y_train_sim), (X_val, y_val, y_val_sim), (X_test, y_test, y_test_sim), features = \
        preprocess_data(df, split_mode="curves")

    # Feature selection
    print("Feature selection...")
    comp_feat = [features.get_loc("current")]
    X_train_tr, X_val_tr, X_test_tr, fs = apply_filter_fs(X_train, X_val, X_test, y_train, k=k, comp_features=comp_feat)
    print("\tFeatures selected:", features[fs.get_support(indices=True)].tolist(), end="\n\n")

    # Data loaders
    data_load_train = data_loader_creation(X_train_tr, y_train, model_type=model_name,
                                           splits=df["test"][(df["test"] != 302) | (df["test"] != 203)].values)
    data_load_val = data_loader_creation(X_val_tr, y_val, model_type=model_name,
                                         splits=df["test"][df["test"] == 203].values, shuffle=False)
    data_load_test = data_loader_creation(X_test_tr, y_test, model_type=model_name,
                                          splits=df["test"][df["test"] == 302].values, shuffle=False)

    # Model
    model, optimizer = initialize_model(model_name, X_train_tr.shape[1], device=device)

    # Train the model
    print("Training model...")
    model, history = train_model(model, {'train': data_load_train, 'val': data_load_val},
                                 MAPE, optimizer, device=device, num_epochs=1000, early_stopping=early_stopping,
                                 patience=50)

    # Test the model
    print("Testing model...")
    y_pred, epoch_loss = model_evaluation(model, data_load_test, MAPE, device=device)

    # Save the model and feature selection filter
    if store:
        print("Saving model...")
        # pickle.dump(fs, open(store_path + "feature_selector_k_" + str(k) + ".pkl", "wb"))
        torch.save(model, store_path + model.__class__.__name__ + "_model_" + str(k) + "_features.pt")
        print("\tModel saved.")

    # Plot training history
    plot_loss(history)


def evaluation_models():
    # Evaluation model parameters
    k = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    print("Loading data...")
    df = pd.read_csv("./Data/Clean_Data_Full.csv")
    print("\tData loaded, shape:", df.shape, end="\n\n")

    # Process the data
    print("Processing data...", end="\n\n")
    (X_train, y_train, y_train_sim), (X_val, y_val, y_val_sim), (X_test, y_test, y_test_sim), features = \
        preprocess_data(df, split_mode="curves")

    # Load feature selection filter
    print("Loading feature selection filter...")
    fs = pickle.load(open("./results/feature_selector_k_" + str(k) + ".pkl", "rb"))

    # Feature selection
    print("Feature selection...")
    comp_feat = [features.get_loc("current")]
    X_train_tr, X_val_tr, X_test_tr, fs = apply_filter_fs(X_train, X_val, X_test, y_train, k=k, fitted_fs=fs,
                                                          comp_features=comp_feat)
    print("\tFeatures selected:", features[fs.get_support(indices=True)].tolist(), end="\n\n")

    # Data loaders
    data_load_test_tab = data_loader_creation(X_test_tr, y_test, model_type="fnn",
                                              splits=df["test"][df["test"] == 302].values, shuffle=False)
    data_load_test_time = data_loader_creation(X_test_tr, y_test, model_type="cnn",
                                               splits=df["test"][df["test"] == 302].values, shuffle=False)

    # Model
    model_FNN = torch.load("./results/FFNN_model_" + str(k) + "_features.pt")
    model_CNN = torch.load("./results/CNN_1D_model_" + str(k) + "_features.pt")
    model_LSTM = torch.load("./results/LSTM_model_" + str(k) + "_features.pt")

    # Test the model
    print("Testing model...")
    y_pred_FNN, epoch_loss_FNN = model_evaluation(model_FNN, data_load_test_tab, MAPE, device=device)
    y_pred_CNN, epoch_loss_CNN = model_evaluation(model_CNN, data_load_test_time, MAPE, device=device)
    y_pred_LSTM, epoch_loss_LSTM = model_evaluation(model_LSTM, data_load_test_time, MAPE, device=device)

    # Format the predictions
    y_sim = y_test_sim.values.reshape(-1)
    y_real = y_test.values.reshape(-1)
    y_pred_FNN = y_pred_FNN.reshape(-1)
    y_pred_CNN = y_pred_CNN.reshape(-1)
    y_pred_LSTM = y_pred_LSTM.reshape(-1) + y_test_sim.values.reshape(-1)

    # Plot the results
    plt.plot(y_real, label="Real", color="black")
    plt.plot(y_sim, label="P2D Model", color="grey")
    plt.plot(y_pred_FNN, label="FNN Model", color="red")
    plt.title("FNN with feature selection")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.grid("on")
    plt.show()

    plt.plot(y_real, label="Real", color="black")
    plt.plot(y_sim, label="P2D Model", color="grey")
    plt.plot(y_pred_LSTM, label="LSTM Model", color="red")
    plt.title("LSTM with feature selection")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.grid("on")
    plt.show()

    plt.plot(y_real, label="Real", color="black")
    plt.plot(y_sim, label="P2D Model", color="grey")
    plt.plot(y_pred_CNN, label="CNN Model", color="red")
    plt.title("CNN with feature selection")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.grid("on")
    plt.show()

    pdf_P2D = np.sort(-y_sim + y_real)
    cdf_P2D = 1. * np.arange(len(pdf_P2D)) / (len(pdf_P2D) - 1)
    pdf_FNN = np.sort(-y_pred_FNN + y_real)
    cdf_FNN = 1. * np.arange(len(pdf_FNN)) / (len(pdf_FNN) - 1)
    pdf_LSTM = np.sort(-y_pred_LSTM + y_real)
    cdf_LSTM = 1. * np.arange(len(pdf_LSTM)) / (len(pdf_LSTM) - 1)
    pdf_CNN = np.sort(-y_pred_CNN + y_real)
    cdf_CNN = 1. * np.arange(len(pdf_CNN)) / (len(pdf_CNN) - 1)
    plt.plot([-0.5, 0, 0.001, 0.5], [0, 0, 1, 1], label="Real", color="black")
    plt.plot(pdf_P2D, cdf_P2D, label="P2D Model", color="grey")
    plt.plot(pdf_FNN, cdf_FNN, label="FNN Model", color="red")
    plt.plot(pdf_LSTM, cdf_LSTM, label="LSTM Model", color="cyan")
    plt.plot(pdf_CNN, cdf_CNN, label="CNN Model", color="green")
    plt.title("CDF of the error")
    plt.legend()
    plt.xlabel("Error")
    plt.ylabel("CDF")
    plt.grid("on")
    plt.show()

    plt.hist(pdf_P2D, bins=100, label="P2D Model", color="grey")
    plt.hist(y_real - y_pred_LSTM, bins=100, label="LSTM Model", color="cyan")
    plt.hist(y_real - y_pred_FNN, bins=100, label="FNN Model", color="red")
    plt.hist(y_real - y_pred_CNN, bins=100, label="CNN Model", color="green")
    plt.title("Histogram of the error")
    plt.legend()
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid("on")
    plt.show()


if __name__ == "__main__":
    # model_training()
    evaluation_models()
    print("Done!")
