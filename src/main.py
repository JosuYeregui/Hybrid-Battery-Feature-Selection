from utils.data_processing import *
from utils.model_utils import *


def main():
    """
    Main function for the application
    """
    # Set up the experiment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "fnn"

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
    X_train_tr, X_val_tr, X_test_tr, fs = apply_filter_fs(X_train, X_val, X_test, y_train, k=20, comp_features=comp_feat)
    print("\tFeatures selected:", features[fs.get_support(indices=True)].tolist(), end="\n\n")

    # Data loaders
    data_load_train = data_loader_creation(X_train_tr, y_train, model_type=model_name,
                                           splits=df["test"][(df["test"] != 302) | (df["test"] != 203)].values)
    data_load_val = data_loader_creation(X_val_tr, y_val, model_type=model_name,
                                         splits=df["test"][df["test"] == 203].values)
    data_load_test = data_loader_creation(X_test_tr, y_test, model_type=model_name,
                                          splits=df["test"][df["test"] == 302].values)

    # Model
    model, optimizer = initialize_model(model_name, X_train_tr.shape[1], device=device)

    # Train the model
    print("Training model...")
    model, history = train_model(model, {'train': data_load_train, 'val': data_load_val},
                                 MAPE, optimizer, device=device)

    # Test the model
    print("Testing model...")
    model_evaluation(model, data_load_test, MAPE, device=device)

    # Plot training history
    plot_loss(history)


if __name__ == "__main__":
    main()
    print("Done!")
