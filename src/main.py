import pickle
import pandas as pd
from utils.data_processing import *
from utils.model_utils import *
from utils.evaluation import *

from matplotlib import gridspec
from scipy import stats


def model_training(model_name='lstm', k=20, early_stopping=True, store=True, store_path="./results/"):
    # Set up the experiment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_ids = list(range(68, 99))
    test_ids = list(range(100, 133))  # list(range(101, 282)) # list(range(282, 292))

    # Load the data
    print("Loading data...")
    df = pd.read_csv("./Data/Data_Final_methods.csv")
    print("\tData loaded, shape:", df.shape, end="\n\n")

    # Process the data
    print("Processing data...", end="\n\n")
    (X_train, y_train, y_train_sim, y_train_FOM), (X_val, y_val, y_val_sim, y_val_FOM), \
        (X_test, y_test, y_test_sim, y_test_FOM), features = \
        preprocess_data(df, split_mode="curves", val_ids=val_ids, test_ids=test_ids)

    # Feature selection
    print("Feature selection...")
    comp_feat = [features.get_loc("I_sim"), features.get_loc("V_Real")]
    X_train_tr, X_val_tr, X_test_tr, fs = apply_filter_fs(X_train, X_val, X_test, y_train, k=k, comp_features=comp_feat,
                                                          meas_func=f_regression)
    print("\tFeatures selected:", features[fs.get_support(indices=True)].tolist(), end="\n\n")

    # Data loaders
    data_load_train = data_loader_creation(X_train_tr, y_train, model_type=model_name,
                                           splits=df["step"][~df["step"].isin([*val_ids, *test_ids])].values)
    data_load_val = data_loader_creation(X_val_tr, y_val, model_type=model_name,
                                         splits=df["step"][df["step"].isin(val_ids)].values, shuffle=False)
    data_load_test = data_loader_creation(X_test_tr, y_test, model_type=model_name,
                                          splits=df["step"][df["step"].isin(test_ids)].values, shuffle=False)

    # Model
    model, optimizer = initialize_model(model_name, X_train_tr.shape[1], device=device)

    # Train the model
    print("Training model...")
    model, history, t_per_epoch = train_model(model, {'train': data_load_train, 'val': data_load_val},
                                 RMSE, optimizer, device=device, num_epochs=1000, early_stopping=early_stopping,
                                 patience=20, compute_time_per_epoch=True)

    # Test the model
    print("Testing model...")
    y_pred, epoch_loss = model_evaluation(model, data_load_test, RMSE, device=device)

    # Save the model and feature selection filter
    if store:
        print("Saving model...")
        pickle.dump(fs, open(store_path + "feature_selector_k_" + str(k) + ".pkl", "wb"))
        torch.save(model, store_path + model.__class__.__name__ + "_model_" + str(k) + "_features.pt")
        print("\tModel saved.")

    # pickle.dump({"compute_time": t_per_epoch, "Acc": epoch_loss}, open(store_path + "feature_comp_k_" + str(k) + ".pkl", "wb"))

    # Plot training history
    plot_loss(history)

    scale = 100.
    y_sim = y_test_sim.values.reshape(-1) * scale
    y_FOM = y_test_FOM.values.reshape(-1) * scale
    y_real = y_test.values.reshape(-1) * scale
    y_pred = y_pred.reshape(-1) * scale
    t = df[df["step"].isin(test_ids)]["t_sim"] / 3600
    plot_curves(t, y_real, y_sim, y_pred, y_FOM, "ML", y_label="SOC")


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'font.sans-serif': 'Times New Roman'})

    # model_training(k=100, model_name="cnn", store=True)

    # k_list = [0, 5, 10, 20, 50, 100, 200, 250]
    # for k in k_list:
    #     print("Training LSTM model with k =", k)
    #     model_training(k=k, model_name="lstm")
    # evaluation_models()

    # k_list = [0, 5, 10, 20, 50, 100, 200, 250]
    # evaluation_features()

    # evaluation_cost_models()

    print("Done!")
