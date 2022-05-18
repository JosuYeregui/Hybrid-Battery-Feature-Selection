import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


def preprocess_data(df, device, split_mode='curves', k=5):
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
        fs = SelectKBest(meas_func, k=k).fit(np.delete(X_train, comp_features, axis=1), y_train)
    else:
        fs = SelectKBest(meas_func, k=k).fit(X_train, y_train)
    # Build mask
    fs_indexes = fs.get_support(indices=True)
    fs_indexes = np.append(fs_indexes, comp_features) if comp_features is not None else fs_indexes
    print("Selected features:", fs_indexes)


# For function testing
if __name__ == "__main__":
    df = pd.read_csv("../Data/Clean_Data_Full.csv")
    (X_train, y_train, y_train_sim), (X_val, y_val, y_val_sim), (X_test, y_test, y_test_sim), features = \
        preprocess_data(df, device="cpu", split_mode="curves", k=5)

    comp_feat = [features.get_loc("current")]
    # Apply feature selection
    apply_filter_fs(X_train, X_val, X_test, y_train, k=5, comp_features=comp_feat)
