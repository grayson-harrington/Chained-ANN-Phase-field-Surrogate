from math import floor
import h5py
import numpy as np

from MLP_Regressor import MLPRegressor, DatasetType

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


def cv_regressor(model_input, autocorrelations, n_folds=1, random_seed=0, regressor_kwargs=None):
    print()
    print("preparing train/test splits")

    if n_folds >= 2:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        cv_splits = kf.split(model_input)
    else:
        inds = np.arange(len(model_input))
        np.random.seed(random_seed)
        np.random.shuffle(inds)
        cutoff = int(len(inds) * 0.85)
        cv_splits = [(inds[:cutoff], inds[cutoff:])]

    # loop through cv_splits and build/test model(s)
    split_index = 0
    cv_model_nmaes = []
    cv_model_nstds = []
    cv_model_r2s = []
    for train_indices, test_indices in cv_splits:
        split_index = split_index + 1
        print(f"\nCV Split: {split_index}\n")
        print("Creating Machine Learning Model(s)")

        # need to get PC scores for both train and test sets
        corrs_train = autocorrelations[train_indices]
        corrs_test = autocorrelations[test_indices]

        pca = PCA(
            svd_solver='full',
            n_components=5,
            random_state=random_seed,
        )
        scores_train = pca.fit_transform(corrs_train.reshape(len(corrs_train), -1))
        scores_test = pca.transform(corrs_test.reshape(len(corrs_test), -1))

        model_output = np.zeros((len(model_input), 5))
        model_output[train_indices] = scores_train
        model_output[test_indices] = scores_test

        # train regressor based on calculated PC scores
        nmae, nstd, r2 = build_regressor(model_input, model_output, train_indices, test_indices, **regressor_kwargs)
        cv_model_nmaes.append(nmae)
        cv_model_nstds.append(nstd)
        cv_model_r2s.append(r2)

    print("\n" * 2)
    print("CV TEST ACCURACY REPORT:")
    print(f"\tnumber of folds:\t\t{n_folds}")
    print(f"\tmean nmae by PC score:\t{np.mean(cv_model_nmaes, axis=0)}")
    print(f"\tmean nstd by PC score:\t{np.mean(cv_model_nstds, axis=0)}")
    print(f"\tmean r2 by PC score:\t{np.mean(cv_model_r2s, axis=0)}")
    print("\n" * 2)


def build_regressor(model_input, model_output, train_indices, test_indices, epochs=200, plot_loss_error=False):
    # create and train model
    nn_model = MLPRegressor(
        model_input=model_input,
        model_output=model_output,
        train_ind=train_indices,
        test_ind=test_indices,
        hidden_shape=(12, 12, 12),
        batch_size=64,  # larger batch size tends to give better results
        dropout_ratio=0.01,
        optimizer_params={"init_lr": 0.0005, "decay": 5e-03},
        loss_func_params={"reduction": "mean", "delta": 1},
        scheduler_params={"n_epochs": n_epochs},
    )

    t_loss, t_err, tst_loss, tst_err = nn_model.fit(n_epochs=epochs)
    if plot_loss_error:
        plot_metrics(nn_model, t_loss, t_err, tst_loss, tst_err)

    nn_model.model_accuracy(
        dataset_type=DatasetType.TRAIN,
        accuracy_measure=mean_absolute_error,
        print_report=True,
        unscale_output=True
    )
    _, nmae, nstd, r2 = nn_model.model_accuracy(
        dataset_type=DatasetType.TEST,
        accuracy_measure=mean_absolute_error,
        print_report=True,
        unscale_output=True
    )

    return nmae, nstd, r2


def plot_metrics(nn_model, t_loss, t_err, tst_loss, tst_err):
    # train/validation loss/accuracy charts
    fig, ax = plt.subplots()
    ax.plot(list(range(n_epochs)), t_loss, label="Training Loss")
    ax.plot(list(range(n_epochs)), tst_loss, label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.plot(list(range(n_epochs)), 1 - t_err, "--", label="Training Accuracy")
    ax2.plot(list(range(n_epochs)), 1 - tst_err, "--", label="Test Accuracy")
    ax2.set_ylabel("Model Accuracy")
    # ax2.set_ylim(bottom=0.7, top=1)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax.set_title("Heterogeneous Loss/Accuracy")

    # plot correlations
    rows = 1
    cols = 5
    fig, axs = plt.subplots(rows, cols)
    for i in range(cols):

        train_y, train_y_pred, r2_train = nn_model.output_correlation(output_index=i,
                                                                      dataset_type=DatasetType.TRAIN)
        test_y, test_y_pred, _ = nn_model.output_correlation(output_index=i, dataset_type=DatasetType.TEST)

        col = i % cols
        row = floor(i / cols)
        axs[col].set_box_aspect(1)
        axs[col].scatter(train_y, train_y_pred, color="red", alpha=0.5, label="Train")
        axs[col].scatter(test_y, test_y_pred, color="blue", alpha=0.5, label="Test")

        mins1 = np.min(np.concatenate((test_y, test_y_pred)), axis=None)
        maxs1 = np.max(np.concatenate((test_y, test_y_pred)), axis=None)
        mins2 = np.min(np.concatenate((train_y, train_y_pred)), axis=None)
        maxs2 = np.max(np.concatenate((train_y, train_y_pred)), axis=None)

        mins = min(mins1, mins2)
        maxs = max(maxs1, maxs2)

        axs[col].plot([mins, maxs], [mins, maxs], "k")
        axs[col].set_xlim(left=mins + mins / 10, right=maxs + maxs / 10)
        axs[col].set_ylim(bottom=mins + mins / 10, top=maxs + maxs / 10)

        axs[col].set_title("PC" + str(i + 1), fontsize=12)

        if i == 0:
            axs[col].legend()

    fig.text(0.5, 0.05, "Truth", ha="center", fontsize=10)
    fig.text(0.05, 0.5, "Prediction", va="center", rotation="vertical", fontsize=10)

    fig.text(0.5, 0.95, "Heterogeneous Phase PC Score Parity Plots", ha="center", fontsize=14)

    plt.show()

    # if save_model:
    #     NN_model.save_model(model_path)


dataset_path = "_datasets/dataset.hdf5"

train_model = True
n_epochs = 200
cv_folds = 6
rnd_seed = 0
plot_progress = False

save_model = False
model_path = "_Regression/_regression_model"

if __name__ == "__main__":
    np.set_printoptions(precision=4)

    # load in test data
    with h5py.File(dataset_path, "r") as f:
        print(list(f.keys()))
        parameters = f["parameters"][...][:, 0:18]
        homonohomo = f["n_phases"][...] - 1
        pc_scores = f["pc_scores"][...][:, :5]
        correlations = f["pc_scores"][...]

    #  remove all single phase samples as this is a heterogeneous model
    hetero = np.where(homonohomo == 1)[0]
    parameters = parameters[hetero]
    pc_scores = pc_scores[hetero]
    correlations = correlations[hetero]

    # do the regressor training / analysis
    if train_model and not save_model:
        # train model with CV and report metrics. Don't save the models produced
        cv_regressor(parameters, correlations, n_folds=cv_folds, random_seed=rnd_seed,
                     regressor_kwargs={"epochs": n_epochs,
                                       "plot_loss_error": plot_progress})
    elif train_model and save_model:
        # train model on random fold and save
        print("TODO")  # TODO?
    elif save_model and not train_model:
        # model = pickle.load(open(model_path + ".p", "rb"))
        #
        # print()
        # print("-" * 25)
        # print("-" * 5 + "Check Model Save" + "-" * 5)
        # print("-" * 25)
        #
        # acc_t, mae_t_mean, mae_t_std = model.model_accuracy(
        #     dataset_type=DatasetType.TRAIN, accuracy_measure=mean_absolute_error, print_report=True
        # )
        # acc_tst, mae_tst_mean, mae_tst_std = model.model_accuracy(
        #     dataset_type=DatasetType.TEST, accuracy_measure=mean_absolute_error, print_report=True
        # )
        #
        # train_scores = np.zeros((len(model.datasets.train.input), 5))
        # train_scores_pred = np.zeros((len(model.datasets.train.input), 5))
        #
        # test_scores = np.zeros((len(model.datasets.test.input), 5))
        # test_scores_pred = np.zeros((len(model.datasets.test.input), 5))
        #
        # # plot correlations
        # rows = 1
        # cols = 5
        # fig, axs = plt.subplots(rows, cols)
        # for i in range(5):
        #
        #     train_y, train_y_pred, _ = model.output_correlation(output_index=i, dataset_type=DatasetType.TRAIN)
        #     test_y, test_y_pred, _ = model.output_correlation(output_index=i, dataset_type=DatasetType.TEST)
        #
        #     train_scores_pred[:, i] = train_y_pred
        #     train_scores[:, i] = train_y
        #     test_scores_pred[:, i] = test_y_pred
        #     test_scores[:, i] = test_y
        #
        #     col = i % cols
        #     row = floor(i / cols)
        #     axs[col].set_box_aspect(1)
        #     axs[col].scatter(train_y, train_y_pred, color="red", alpha=0.5, label="Train")
        #     axs[col].scatter(test_y, test_y_pred, color="blue", alpha=0.5, label="Test")
        #
        #     mins1 = np.min(np.concatenate((test_y[:, i], test_y_pred[:, i])), axis=None)
        #     maxs1 = np.max(np.concatenate((test_y[:, i], test_y_pred[:, i])), axis=None)
        #     mins2 = np.min(np.concatenate((train_y[:, i], train_y_pred[:, i])), axis=None)
        #     maxs2 = np.max(np.concatenate((train_y[:, i], train_y_pred[:, i])), axis=None)
        #
        #     mins = min(mins1, mins2)
        #     maxs = max(maxs1, maxs2)
        #
        #     axs[col].plot([mins, maxs], [mins, maxs], "k")
        #     axs[col].set_xlim(left=mins + mins / 10, right=maxs + maxs / 10)
        #     axs[col].set_ylim(bottom=mins + mins / 10, top=maxs + maxs / 10)
        #
        #     axs[col].set_title("PC" + str(i + 1), fontsize=12)
        #
        #     if i == 0:
        #         axs[col].legend()
        #     if row != rows - 1:
        #         axs[col].tick_params(bottom=False)
        #         axs[col].tick_params(labelbottom=False)
        #     if col != 0:
        #         axs[col].tick_params(left=False)
        #         axs[col].tick_params(labelleft=False)
        #
        # fig.text(0.5, 0.05, "Truth", ha="center", fontsize=10)
        # fig.text(0.05, 0.5, "Prediction", va="center", rotation="vertical", fontsize=10)
        #
        # fig.text(
        #     0.5,
        #     0.95,
        #     "Heterogeneous Phase PC Score Parity Plots",
        #     ha="center",
        #     fontsize=14,
        # )
        #
        # # plot error histograms
        # fig2, axs2 = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True)
        #
        # nbins_t = 20
        # nbins_v = 10
        # nbins_tst = 15
        #
        # bin_w = 0.05
        # bins = np.arange(0, 0.5 + bin_w, bin_w)
        #
        # print(len(acc_t))
        # print(acc_t.shape)
        # # axs2[0].hist(acc_t[:, 0], bins=bins)
        # axs2[0].hist(acc_tst[:, 0], bins=bins)  # , color="tab:orange")
        # # axs2[0].hist(acc_v[:, 0], bins=bins)
        # axs2[0].set_title("PC1 Error", fontsize=12)
        #
        # # axs2[1].hist(acc_t[:, 1], bins=bins)
        # axs2[1].hist(acc_tst[:, 1], bins=bins)  # , color="tab:orange")
        # # axs2[1].hist(acc_v[:, 1], bins=bins)
        # axs2[1].set_title("PC2 Error", fontsize=12)
        #
        # # axs2[2].hist(acc_t[:, 2], bins=bins)
        # axs2[2].hist(acc_tst[:, 2], bins=bins)  # , color="tab:orange")
        # # axs2[2].hist(acc_v[:, 2], bins=bins)
        # axs2[2].set_title("PC3 Error", fontsize=12)
        #
        # # axs2[3].hist(acc_t[:, 3], bins=bins)
        # axs2[3].hist(acc_tst[:, 3], bins=bins)  # , color="tab:orange")
        # # axs2[3].hist(acc_v[:, 3], bins=bins)
        # axs2[3].set_title("PC4 Error", fontsize=12)
        #
        # # axs2[4].hist(acc_t[:, 4], bins=bins)
        # axs2[4].hist(acc_tst[:, 4], bins=bins)  # , color="tab:orange")
        # # axs2[4].hist(acc_v[:, 4], bins=bins)
        # axs2[4].set_title("PC5 Error", fontsize=12)
        #
        # axs2[0].set_ylabel("Frequency")
        # axs2[2].set_xlabel("Normalized Absolute Error")
        #
        # plt.show()
        print("TODO")  # TODO
