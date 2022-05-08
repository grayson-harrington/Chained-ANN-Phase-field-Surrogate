from math import floor, ceil
import h5py
import numpy as np

from MLP_Regressor import MLPRegressor, DatasetType

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


def cv_regressor(model_input, autocorrelations, n_folds=1, random_seed=0, save_tests=False, regressor_kwargs=None):
    print()
    print("preparing train/test splits")

    if n_folds >= 2:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        cv_splits = kf.split(model_input)
    else:
        inds = np.arange(len(model_input))
        np.random.seed(random_seed)
        np.random.shuffle(inds)
        cutoff = int(len(inds) * 0.9)
        cv_splits = [(inds[:cutoff], inds[cutoff:])]

    # loop through cv_splits and build/test model(s)
    cv_model_nmaes_t = []
    cv_model_nstds_t = []
    cv_model_r2s_t = []
    cv_model_nmaes_tst = []
    cv_model_nstds_tst = []
    cv_model_r2s_tst = []

    split_index = 0
    for train_indices, test_indices in cv_splits:
        split_index = split_index + 1
        print(f"\nCV Split: {split_index}\n")

        # need to get PC scores for both train and test sets
        corrs_train = autocorrelations[train_indices]
        corrs_test = autocorrelations[test_indices]

        print("getting PC scores from autocorrelations")
        pca = PCA(
            svd_solver='full',
            n_components=5,
            random_state=random_seed,
        )
        scores_train = pca.fit_transform(corrs_train.reshape(len(corrs_train), -1))
        scores_test = pca.transform(corrs_test.reshape(len(corrs_test), -1))

        # Use PC scores to train and test model
        model_output = np.zeros((len(model_input), 5))
        model_output[train_indices] = scores_train
        model_output[test_indices] = scores_test

        # train regressor based on calculated PC scores
        print("training model")
        nn_model = build_regressor(model_input, model_output, train_indices, test_indices, **regressor_kwargs)

        y_t, y_pred_t, nmae_t, nstd_t, r2_t = nn_model.model_accuracy(
            dataset_type=DatasetType.TRAIN,
            print_report=True,
            unscale_output=True
        )
        y_tst, y_pred_tst, nmae_tst, nstd_tst, r2_tst = nn_model.model_accuracy(
            dataset_type=DatasetType.TEST,
            print_report=True,
            unscale_output=True
        )

        # transform predicted pc scores back to autocorrelations
        print("reconstructing autocorrelations from PC scores")
        corrs_test_retained = []
        for scores in scores_test:
            corr = np.dot(scores, pca.components_) + pca.mean_
            corr = corr.reshape(corrs_test[0].shape)
            corrs_test_retained.append(corr)

        corrs_test_pred = []
        for scores in y_pred_tst:
            corr = np.dot(scores, pca.components_) + pca.mean_
            corr = corr.reshape(corrs_test[0].shape)
            corrs_test_pred.append(corr)

        print("saving autocorrelations")
        if save_tests:
            with h5py.File(f"_Regression/{n_folds}-fold reconsucted_statistics.hdf5", 'a') as f:
                grp = f.create_group(f"cv{split_index}")
                grp.create_dataset("corrs_test_retained", data=corrs_test_retained, compression="gzip")
                grp.create_dataset("corrs_test_pred", data=corrs_test_pred, compression="gzip")
                grp.create_dataset("corrs_test_full", data=corrs_test, compression="gzip")

        cv_model_nmaes_t.append(nmae_t)
        cv_model_nstds_t.append(nstd_t)
        cv_model_r2s_t.append(r2_t)
        cv_model_nmaes_tst.append(nmae_tst)
        cv_model_nstds_tst.append(nstd_tst)
        cv_model_r2s_tst.append(r2_tst)

    print("\n" * 2)
    print("CV TRAIN ACCURACY REPORT:")
    print(f"\tnumber of folds:\t\t{n_folds}")
    print(f"\tmean nmae by PC score:\t{np.mean(cv_model_nmaes_t, axis=0)}")
    print(f"\tmean nstd by PC score:\t{np.mean(cv_model_nstds_t, axis=0)}")
    print(f"\tmean r2 by PC score:\t{np.mean(cv_model_r2s_t, axis=0)}")
    print("\n" * 2)
    print("CV TEST ACCURACY REPORT:")
    print(f"\tnumber of folds:\t\t{n_folds}")
    print(f"\tmean nmae by PC score:\t{np.mean(cv_model_nmaes_tst, axis=0)}")
    print(f"\tmean nstd by PC score:\t{np.mean(cv_model_nstds_tst, axis=0)}")
    print(f"\tmean r2 by PC score:\t{np.mean(cv_model_r2s_tst, axis=0)}")
    print("\n" * 2)


def build_regressor(model_input, model_output, train_indices, test_indices, epochs=200,
                    print_metrics=False, plot_loss_error=False):
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

    if print_metrics:
        nn_model.model_accuracy(
            dataset_type=DatasetType.TRAIN,
            print_report=True,
            unscale_output=True
        )
        nn_model.model_accuracy(
            dataset_type=DatasetType.TEST,
            print_report=True,
            unscale_output=True
        )

    return nn_model


def plot_metrics(nn_model, t_loss, t_err, tst_loss, tst_err):
    epochs = len(t_loss)
    # train/validation loss/accuracy charts
    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), t_loss, label="Training Loss")
    ax.plot(list(range(epochs)), tst_loss, label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.plot(list(range(epochs)), 1 - t_err, "--", label="Training Accuracy")
    ax2.plot(list(range(epochs)), 1 - tst_err, "--", label="Test Accuracy")
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


dataset_path = "_datasets/dataset.hdf5"

n_epochs = 200
cv_folds = 1  # 10 for paper
rnd_seed = 0
plot_progress = False
save_cv_tests = False

if __name__ == "__main__":
    np.set_printoptions(precision=4)

    # load in data
    with h5py.File(dataset_path, "r") as f:
        print(list(f.keys()))
        parameters = f["parameters"][...][:, 0:18]
        homonohomo = f["n_phases"][...] - 1
        correlations = f["correlations"][...]

    #  remove all single phase samples as this is a heterogeneous model
    hetero = np.where(homonohomo == 1)[0]
    parameters = parameters[hetero]
    correlations = correlations[hetero]

    # train model with CV and report metrics. Don't save the models produced
    cv_regressor(parameters, correlations, n_folds=cv_folds, random_seed=rnd_seed, save_tests=save_cv_tests,
                 regressor_kwargs={"epochs": n_epochs,
                                   "plot_loss_error": plot_progress})
