import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import torch
from torch.utils.data import TensorDataset, DataLoader


def preprocess_data(df, split_mode='curves'):
    """
    Processes the dataframe to remove unnecessary columns and
    apply preprocessing techniques.
    """
    # Split phase taking into account the split mode
    if split_mode == 'sample':
        # Split dataframe into train, validation and test
        (X_train, y_train, y_train_sim), (X_val, y_val, y_val_sim), (X_test, y_test, y_test_sim) = \
            [(x.drop(columns=["E_real", "Ecell", "time", "split", "test"]), x["E_real"], x["Ecell"])
             for _, x in df.groupby(df['split'])]
    elif split_mode == 'curves':
        # Split data into training, validation and test sets
        # Training set
        X_train = df[(df["test"] != 302) | (df["test"] != 203)]
        y_train = X_train["E_real"]
        y_train_sim = X_train["Ecell"]
        X_train = X_train.drop(columns=["E_real", "Ecell", "time", "split", "test"])
        # Validation set
        X_val = df[df["test"] == 203]
        y_val = X_val["E_real"]
        y_val_sim = X_val["Ecell"]
        X_val = X_val.drop(columns=["E_real", "Ecell", "time", "split", "test"])
        # Test set
        X_test = df[df["test"] == 302]
        y_test = X_test["E_real"]
        y_test_sim = X_test["Ecell"]
        X_test = X_test.drop(columns=["E_real", "Ecell", "time", "split", "test"])
    else:
        raise ValueError("Invalid split mode")
    # Get feature column names
    x_feats = X_train.columns

    # Preprocessing phase
    # Normalize data
    scaler = preprocessing.StandardScaler().fit(X_train)
    # Apply scaling
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train, y_train_sim), (X_val, y_val, y_val_sim), (X_test, y_test, y_test_sim), x_feats


def apply_filter_fs(X_train, X_val, X_test, y_train, k=5, fitted_fs=None,
                    meas_func=mutual_info_regression, comp_features=None):
    """
    Applies feature selection to the data and returns the k_best.
    """
    # Feature selection
    k = k if k < X_train.shape[1] else X_train.shape[1]
    # Define selector
    if fitted_fs is not None:
        fs = fitted_fs
        fs.k = k
    elif comp_features is not None:
        k -= len(comp_features)
        X_train_empt = X_train.copy()
        X_train_empt[:, comp_features] = 0  # Column to 0 to ensure that the features are not selected
        fs = SelectKBest(meas_func, k=k).fit(X_train_empt, y_train)
    else:
        fs = SelectKBest(meas_func, k=k).fit(X_train, y_train)
    # Build mask
    fs_indexes = fs.get_support(indices=True)
    fs_indexes = np.append(fs_indexes, comp_features) if comp_features is not None else fs_indexes
    # Transform data
    X_train = X_train[:, fs_indexes]
    X_val = X_val[:, fs_indexes]
    X_test = X_test[:, fs_indexes]
    # Return transformed data
    return X_train, X_val, X_test, fs


def data_loader_creation(X, y, device, batch_size=32, shuffle=True, model_type='fnn'):
    """
    Creates a data loader for the given data.
    """
    if model_type == 'fnn':
        # Create dataset
        dataset = TensorDataset(torch.from_numpy(X).float().to(device), torch.from_numpy(y).float().to(device))
        # Create data loader
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    elif model_type == 'lstm':
        # Create data loader
        data_loader = DataLoader(
            dataset=TensorDataset(torch.from_numpy(X).float().to(device),
                                  torch.from_numpy(y).float().to(device)),
            batch_size=batch_size,
            shuffle=shuffle
        )
    elif model_type == 'cnn':
        # Create data loader
        data_loader = DataLoader(
            dataset=TensorDataset(torch.from_numpy(X).float().to(device),
                                  torch.from_numpy(y).float().to(device)),
            batch_size=batch_size,
            shuffle=shuffle
        )
    else:
        raise ValueError("Invalid model type")

    # Return data loader
    return data_loader


# For function testing
if __name__ == "__main__":
    df = pd.read_csv("../Data/Clean_Data_Full.csv")
    (X_train, y_train, y_train_sim), (X_val, y_val, y_val_sim), (X_test, y_test, y_test_sim), features = \
        preprocess_data(df, split_mode="curves")

    comp_feat = [features.get_loc("current")]
    # Execution time check
    # Apply feature selection
    print("Applying feature selection")
    X_train_tr, X_val_tr, X_test_tr, fs = apply_filter_fs(X_train, X_val, X_test, y_train, k=5, comp_features=comp_feat)
    print("Feature selection done")
    X_train_tr, X_val_tr, X_test_tr, fs = apply_filter_fs(X_train, X_val, X_test, y_train,
                                                          fitted_fs=fs, k=10, comp_features=comp_feat)
    print("Feature selection done")
