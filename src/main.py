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
    model_name = "cnn"
    store_path = "./results/"
    store = True
    early_stopping = True
    k = 20
    val_ids = [203]
    test_ids = [302]

    # Load the data
    print("Loading data...")
    df = pd.read_csv("./Data/Clean_Data_Full.csv")
    print("\tData loaded, shape:", df.shape, end="\n\n")

    # Process the data
    print("Processing data...", end="\n\n")
    (X_train, y_train, y_train_sim), (X_val, y_val, y_val_sim), (X_test, y_test, y_test_sim), features = \
        preprocess_data(df, split_mode="curves", val_ids=val_ids, test_ids=test_ids)

    # Feature selection
    print("Feature selection...")
    comp_feat = [features.get_loc("current")]
    X_train_tr, X_val_tr, X_test_tr, fs = apply_filter_fs(X_train, X_val, X_test, y_train, k=k, comp_features=comp_feat)
    print("\tFeatures selected:", features[fs.get_support(indices=True)].tolist(), end="\n\n")

    # Data loaders
    data_load_train = data_loader_creation(X_train_tr, y_train, model_type=model_name,
                                           splits=df["test"][~df["test"].isin([*val_ids, *test_ids])].values)
    data_load_val = data_loader_creation(X_val_tr, y_val, model_type=model_name,
                                         splits=df["test"][df["test"].isin(val_ids)].values, shuffle=False)
    data_load_test = data_loader_creation(X_test_tr, y_test, model_type=model_name,
                                          splits=df["test"][df["test"].isin(test_ids)].values, shuffle=False)

    # Model
    model, optimizer = initialize_model(model_name, X_train_tr.shape[1], device=device)

    # Train the model
    print("Training model...")
    model, history = train_model(model, {'train': data_load_train, 'val': data_load_val},
                                 MAPE, optimizer, device=device, num_epochs=1000, early_stopping=early_stopping,
                                 patience=200)

    # Test the model
    print("Testing model...")
    y_pred, epoch_loss = model_evaluation(model, data_load_test, MAPE, device=device)

    # Save the model and feature selection filter
    if store:
        print("Saving model...")
        pickle.dump(fs, open(store_path + "feature_selector_k_" + str(k) + ".pkl", "wb"))
        torch.save(model, store_path + model.__class__.__name__ + "_model_" + str(k) + "_features.pt")
        print("\tModel saved.")

    # Plot training history
    plot_loss(history)


def evaluation_models():
    # Evaluation model parameters
    k = 20
    test_ids = [302]
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
                                              splits=df["test"][df["test"].isin(test_ids)].values, shuffle=False)
    data_load_test_time = data_loader_creation(X_test_tr, y_test, model_type="cnn",
                                               splits=df["test"][df["test"].isin(test_ids)].values, shuffle=False)

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
    y_pred_LSTM = y_pred_LSTM.reshape(-1)
    t = df[df["test"] == 302]["time"]/3600

    # Plot the results
    # Prediction curves
    plot_curves(t, y_real, y_sim, y_pred_FNN, "FNN")
    plot_curves(t, y_real, y_sim, y_pred_CNN, "CNN")
    plot_curves(t, y_real, y_sim, y_pred_LSTM, "LSTM")
    # CDF curves
    iter_list = [("P2D Model", y_sim, "grey"), ("FNN Model", y_pred_FNN, "red"), ("CNN Model", y_pred_CNN, "blue"),
                 ("LSTM Model", y_pred_LSTM, "green")]
    cdf_dict = {"Reference": {"pdf": np.array([0, 0, 0.5]), "cdf": np.array([0, 1, 1]), "color": "black", "linestyle": "--"}}
    for item in iter_list:
        pdf = np.sort(np.abs(y_real - item[1]))
        cdf = 1. * np.arange(len(pdf)) / (len(pdf) - 1)

        cdf_dict[item[0]] = {"pdf": pdf, "cdf": cdf, "color": item[2], "linestyle": "-"}
    plot_cdf(cdf_dict)


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif', 'font.sans-serif': 'Times New Roman'})
    #model_training()
    evaluation_models()
    print("Done!")
