import pickle
import pandas as pd
from .data_processing import *
from .model_utils import *


def evaluation_models():
    # Evaluation model parameters
    k = 20
    test_ids = list(range(100, 133))  # list(range(68, 99))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    print("Loading data...")
    df = pd.read_csv("./Data/Data_Final_methods.csv")
    print("\tData loaded, shape:", df.shape, end="\n\n")

    # Process the data
    print("Processing data...", end="\n\n")
    (X_train, y_train, y_train_sim, y_train_FOM), (X_val, y_val, y_val_sim, y_val_FOM), \
        (X_test, y_test, y_test_sim, y_test_FOM), features = \
        preprocess_data(df, split_mode="curves")

    # Load feature selection filter
    print("Loading feature selection filter...")
    fs = pickle.load(open("./results/feature_selector_k_" + str(k) + ".pkl", "rb"))

    # Feature selection
    print("Feature selection...")
    comp_feat = [features.get_loc("I_sim"), features.get_loc("V_Real")]
    X_train_tr, X_val_tr, X_test_tr, fs = apply_filter_fs(X_train, X_val, X_test, y_train, k=k, fitted_fs=fs,
                                                          comp_features=comp_feat)
    print("\tFeatures selected:", features[fs.get_support(indices=True)].tolist(), end="\n\n")

    # Data loaders
    data_load_test_tab = data_loader_creation(X_test_tr, y_test, model_type="fnn",
                                              splits=df["step"][df["step"].isin(test_ids)].values, shuffle=False)
    data_load_test_time = data_loader_creation(X_test_tr, y_test, model_type="cnn",
                                               splits=df["step"][df["step"].isin(test_ids)].values, shuffle=False)

    # Model
    model_FNN = torch.load("./results/FFNN_model_" + str(k) + "_features.pt")
    model_CNN = torch.load("./results/CNN_1D_model_" + str(k) + "_features.pt")
    model_LSTM = torch.load("./results/LSTM_model_" + str(k) + "_features.pt")

    # Test the model
    print("Testing model...")
    y_pred_FNN, epoch_loss_FNN = model_evaluation(model_FNN, data_load_test_tab, RMSE, device=device)
    y_pred_CNN, epoch_loss_CNN = model_evaluation(model_CNN, data_load_test_time, RMSE, device=device)
    y_pred_LSTM, epoch_loss_LSTM = model_evaluation(model_LSTM, data_load_test_time, RMSE, device=device)

    # Format the predictions
    scale = 100.
    y_sim = y_test_sim.values.reshape(-1) * scale
    y_FOM = y_test_FOM.values.reshape(-1) * scale
    y_real = y_test.values.reshape(-1) * scale
    print("ROM: RMSE: " + str(np.sqrt(np.mean((y_sim-y_real)**2))) +
          " MAE: " + str(np.sum(np.abs(y_sim-y_real))/len(y_sim)) +
          " R_sq: " + str(1. - np.sum((y_sim-y_real)**2)/np.sum((y_sim-np.mean(y_real))**2)))
    print("FOM: RMSE: " + str(np.sqrt(np.mean((y_FOM - y_real) ** 2))) +
          " MAE: " + str(np.sum(np.abs(y_FOM - y_real)) / len(y_FOM)) +
          " R_sq: " + str(1. - np.sum((y_FOM - y_real) ** 2) / np.sum((y_FOM - np.mean(y_real)) ** 2)))
    y_pred_FNN = y_pred_FNN.reshape(-1) * scale
    y_pred_CNN = y_pred_CNN.reshape(-1) * scale
    y_pred_LSTM = y_pred_LSTM.reshape(-1) * scale
    t = df[df["step"].isin(test_ids)]["t_sim"].to_numpy()/3600.
    t = t - t[0]

    print("FNN: RMSE: " + str(np.sqrt(np.mean((y_pred_FNN - y_real) ** 2))) +
          " MAE: " + str(np.sum(np.abs(y_pred_FNN - y_real)) / len(y_pred_FNN)) +
          " R_sq: " + str(1. - np.sum((y_pred_FNN - y_real) ** 2) / np.sum((y_pred_FNN - np.mean(y_real)) ** 2)))
    print("CNN: RMSE: " + str(np.sqrt(np.mean((y_pred_CNN - y_real) ** 2))) +
          " MAE: " + str(np.sum(np.abs(y_pred_CNN - y_real)) / len(y_pred_CNN)) +
          " R_sq: " + str(1. - np.sum((y_pred_CNN - y_real) ** 2) / np.sum((y_pred_CNN - np.mean(y_real)) ** 2)))
    print("LSTM: RMSE: " + str(np.sqrt(np.mean((y_pred_LSTM - y_real) ** 2))) +
          " MAE: " + str(np.sum(np.abs(y_pred_LSTM - y_real)) / len(y_pred_LSTM)) +
          " R_sq: " + str(1. - np.sum((y_pred_LSTM - y_real) ** 2) / np.sum((y_pred_LSTM - np.mean(y_real)) ** 2)))

    # Plot the results
    # Prediction curves
    plot_curves(t, y_real, y_sim, y_pred_FNN, y_FOM, "FNN", y_label="SOC [\%]", save=True, filename="Outputs/FNN_"+str(k)+"feats_models")
    y_pred_CNN[0:10] = y_pred_CNN[10]
    plot_curves(t, y_real, y_sim, y_pred_CNN, y_FOM, "CNN", y_label="SOC [\%]", save=True, filename="Outputs/CNN_"+str(k)+"feats_models")
    y_pred_LSTM[0:10] = y_pred_LSTM[10]
    plot_curves(t, y_real, y_sim, y_pred_LSTM, y_FOM, "LSTM", y_label="SOC [\%]", save=True, filename="Outputs/LSTM_"+str(k)+"feats_models")
    # CDF curves
    iter_list = [("ROM Model", y_sim, "#004488"), ("FOM Model", y_FOM, "springgreen"), ("Seq. FNN Model", y_pred_FNN, "#AA3377"),
                 ("Seq. CNN Model", y_pred_CNN, "#DDAA33"), ("Seq. LSTM Model", y_pred_LSTM, "#E9002D")]
    cdf_dict = {}
    for item in iter_list:
        pdf = np.sort(np.abs(y_real - item[1])/scale)
        cdf = 1. * np.arange(len(pdf)) / (len(pdf) - 1)
        # add to dictionary
        cdf_dict[item[0]] = {"pdf": pdf, "cdf": cdf, "color": item[2], "linestyle": "-"}
    plot_cdf(cdf_dict, xlabel="Absolute SOC error [\%]", ylabel="CDF", scaling=100, save=True, filename="Outputs/cdf_"+str(k)+"feats_models")


def evaluation_features(save=True):
    # Evaluation model parameters
    with plt.style.context(['science', 'high-contrast']):
        cm = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(dpi=600.)

        k_list = [0, 5, 50, 100, 200, 250]
        l = []
        acc = []
        for k in k_list:
            with open("results/feature_comp_k_" + str(k) + ".pkl", 'rb') as pickle_file:
                f = pickle.load(pickle_file)
                l.append(f["compute_time"])
                acc.append(f["Acc"] * 100)

        k_list[4] = 175

        ax_r = ax.twinx()

        ax_r.plot(k_list, acc, '--', color='royalblue')
        ax_r.set_ylim(0, 1.4)

        bplot = ax.boxplot(l, autorange=True, manage_ticks=False, positions=k_list,
                           showfliers=False, widths=3, patch_artist=True)

        colors = ['#FEBE00'] + ['#E9002D'] * (len(k_list) - 1)

        for patch, median, color in zip(bplot['boxes'], bplot['medians'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#3A3B3C")
            patch.set_lw(1.)
            patch.set_alpha(0.5)
            median.set_color('#3A3B3C')
            median.set_lw(1.)

        ax.grid("major")

        ax.set_xlabel("k selected features")
        ax.set_ylabel("Time per epoch [s]")
        ax_r.set_ylabel("RMSE")
        ax_r.yaxis.label.set_color('royalblue')
        [t.set_color('royalblue') for t in ax_r.yaxis.get_ticklines()]
        [t.set_color('royalblue') for t in ax_r.yaxis.get_ticklabels()]

        ax.legend(bplot["boxes"], ["LSTM Model", "Seq. LSTM Models"], loc="lower right", frameon=True, fancybox=True,
                  framealpha=1., fontsize=7)

        if save:
            fig.savefig("Outputs/feature_selection_eval.pdf")
        plt.show()


def evaluation_cost_models():
    # Evaluation model parameters
    k = 250
    test_ids = list(range(100, 133))  # list(range(68, 99))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    print("Loading data...")
    df = pd.read_csv("./Data/Data_Final_methods.csv")
    print("\tData loaded, shape:", df.shape, end="\n\n")

    dt_sim_FOM_ps = [0.220820805316457, 0.220313250042194, 0.221655864135022, 0.222097834177214, 0.225787981518986,
                    0.222223518059073, 0.216080432742615, 0.219712702025317, 0.224957719409283, 0.220190877215190,
                    0.223808790717300, 0.221507246244727, 0.222686115358649, 0.221412052742615, 0.222768893670887,
                    0.217577436202532, 0.220797359409283, 0.218470737890294, 0.220028256455697, 0.219227136202533,
                    0.215509240253164, 0.220466280421940, 0.228281667848101, 0.220570725654007, 0.208595404725736,
                    0.229795671561179, 0.220788112911393, 0.224408945063290, 0.224172094430380, 0.220378363544305,
                    0.217731259156118, 0.222906311392406, 0.222598720675107]
    dt_sim_ROM = df[df["step"].isin(test_ids)]["dt_sim_ROM"]
    steps = df[df["step"].isin(test_ids)]["step"]

    dt_FOM_dict = {u: t for u, t in zip(np.unique(steps), dt_sim_FOM_ps)}
    dt_sim_FOM = np.array([dt_FOM_dict[s] for s in steps])

    # Process the data
    print("Processing data...", end="\n\n")
    (X_train, y_train, y_train_sim, y_train_FOM), (X_val, y_val, y_val_sim, y_val_FOM), \
        (X_test, y_test, y_test_sim, y_test_FOM), features = \
        preprocess_data(df, split_mode="curves")

    # Load feature selection filter
    print("Loading feature selection filter...")
    fs = pickle.load(open("./results/feature_selector_k_" + str(k) + ".pkl", "rb"))

    # Feature selection
    print("Feature selection...")
    comp_feat = [features.get_loc("I_sim"), features.get_loc("V_Real")]
    X_train_tr, X_val_tr, X_test_tr, fs = apply_filter_fs(X_train, X_val, X_test, y_train, k=k, fitted_fs=fs,
                                                          comp_features=comp_feat)
    print("\tFeatures selected:", features[fs.get_support(indices=True)].tolist(), end="\n\n")

    # Data loaders
    data_load_test_time = data_loader_creation(X_test_tr, y_test, model_type="lstm",
                                               splits=df["step"][df["step"].isin(test_ids)].values, shuffle=False)

    # Model
    model_LSTM = torch.load("./results/LSTM_model_" + str(k) + "_features.pt").to(device)

    # Test the model
    print("Testing model...")
    y_pred_LSTM, epoch_loss_LSTM, exec_time = \
        model_evaluation(model_LSTM, data_load_test_time, RMSE, device=device, return_exec_time=True)

    dt_sim_model = dt_sim_ROM + exec_time

    # k=0
    # Process the data
    print("Processing data...", end="\n\n")
    (X_train, y_train, y_train_sim, y_train_FOM), (X_val, y_val, y_val_sim, y_val_FOM), \
    (X_test, y_test, y_test_sim, y_test_FOM), features = \
        preprocess_data(df, split_mode="curves")

    # Feature selection
    print("Feature selection...")
    comp_feat = [features.get_loc("I_sim"), features.get_loc("V_Real")]
    X_train_tr, X_val_tr, X_test_tr, fs = apply_filter_fs(X_train, X_val, X_test, y_train, k=0,
                                                          comp_features=comp_feat)
    print("\tFeatures selected:", features[fs.get_support(indices=True)].tolist(), end="\n\n")

    # Data loaders
    data_load_test_time = data_loader_creation(X_test_tr, y_test, model_type="lstm",
                                               splits=df["step"][df["step"].isin(test_ids)].values, shuffle=False)

    # Model
    model_LSTM_k0 = torch.load("./results/LSTM_model_0_features.pt").to(device)

    # Test the model
    print("Testing model...")
    y_pred_LSTM_k0, epoch_loss_LSTM_k0, exec_time_k0 = \
        model_evaluation(model_LSTM_k0, data_load_test_time, RMSE, device=device, return_exec_time=True)

    scale = 100.
    y_ROM = y_test_sim.values.reshape(-1) * scale
    y_FOM = y_test_FOM.values.reshape(-1) * scale
    y_real = y_test.values.reshape(-1) * scale
    y_pred_LSTM = y_pred_LSTM.reshape(-1) * scale
    y_pred_LSTM[0:2] = y_pred_LSTM[2]
    y_pred_LSTM_k0 = y_pred_LSTM_k0.reshape(-1) * scale
    y_pred_LSTM_k0[0:2] = y_pred_LSTM_k0[2]

    plot_dict = {"Real": y_real, "ROM": [dt_sim_ROM, y_ROM],
                 "FOM": [dt_sim_FOM, y_FOM], "RawML": [exec_time_k0, y_pred_LSTM_k0], "LSTM": [dt_sim_model, y_pred_LSTM]}

    plot_time_models(plot_dict, save=True, filename="Outputs/cost_accuracy")
