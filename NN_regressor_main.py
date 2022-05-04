import math
import pickle
import numpy as np
import h5py

from MLP_Regressor import MLPRegressor, DatasetType

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

n_epochs = 200
n_pc_scores = 5

train_model = True
save_model = True
model_path = "_Regression/_regression_model"


def main():
    print()
    print("Creating Machine Learning Model")

    NN_model = MLPRegressor(
        train_location="datasets/train.hdf5",
        validate_location="datasets/validate.hdf5",
        test_location="datasets/test.hdf5",
        hidden_shape=(12, 12, 12),
        batch_size=64,  # larger batch size tends to give better results
        dropout_ratio=0.01,
        optimizer_params={"init_lr": 0.0005, "decay": 5e-03},
        loss_func_params={"reduction": "mean", "delta": 1},
        scheduler_params={"n_epochs": n_epochs},
    )

    t_loss, t_err, v_loss, v_err = NN_model.fit(n_epochs=n_epochs)

    _, mae_t_mean, mae_t_std = NN_model.model_accuracy(
        dataset_type=DatasetType.TRAIN, accuracy_measure=mean_absolute_error
    )
    _, mae_v_mean, mae_v_std = NN_model.model_accuracy(
        dataset_type=DatasetType.VALIDATE, accuracy_measure=mean_absolute_error
    )

    if save_model:
        NN_model.save_model(model_path)

    print()
    print("Train NMAE: %.4f" % mae_t_mean)
    print("Validate NMAE: %.4f" % mae_v_mean)
    print()
    print("Train STDNAE: %.4f" % mae_t_std)
    print("Validate STDNAE: %.4f" % mae_v_std)

    # train/validation loss/accuracy charts
    fig, ax = plt.subplots()
    ax.plot(list(range(n_epochs)), t_loss, label="Training Loss")
    ax.plot(list(range(n_epochs)), v_loss, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.plot(list(range(n_epochs)), 1 - t_err, "--", label="Training Accuracy")
    ax2.plot(list(range(n_epochs)), 1 - v_err, "--", label="Validation Accuracy")
    ax2.set_ylabel("Model Accuracy")
    ax2.set_ylim(bottom=0.7, top=1)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax.set_title("Heterogeneous Loss/Accuracy")

    # plot correlations
    rows = 1
    cols = n_pc_scores
    fig, axs = plt.subplots(rows, cols)
    for i in range(n_pc_scores):

        train_y, train_y_pred, r2_train = NN_model.output_correlation(
            output_index=i, dataset_type=DatasetType.TRAIN
        )
        validate_y, validate_y_pred, _ = NN_model.output_correlation(
            output_index=i, dataset_type=DatasetType.VALIDATE
        )

        col = i % cols
        row = math.floor(i / cols)
        axs[col].set_box_aspect(1)
        axs[col].scatter(
            train_y, train_y_pred, color="red", alpha=0.5, label="Training"
        )
        axs[col].scatter(
            validate_y, validate_y_pred, color="blue", alpha=0.5, label="Validation"
        )
        axs[col].plot([-1, 1], [-1, 1], "k")

        axs[col].set_xlim(left=-1.1, right=1.1)
        axs[col].set_ylim(bottom=-1.1, top=1.1)

        axs[col].set_title("PC" + str(i + 1), fontsize=12)

        if i == 0:
            axs[col].legend()
        if row != rows - 1:
            axs[col].tick_params(bottom=False)
            axs[col].tick_params(labelbottom=False)
        if col != 0:
            axs[col].tick_params(left=False)
            axs[col].tick_params(labelleft=False)

    fig.text(0.5, 0.05, "Truth", ha="center", fontsize=10)
    fig.text(0.05, 0.5, "Prediction", va="center", rotation="vertical", fontsize=10)

    fig.text(
        0.5, 0.95, "Heterogeneous Phase PC Score Parity Plots", ha="center", fontsize=14
    )

    # plt.show()


if __name__ == "__main__":

    np.set_printoptions(precision=4)

    if train_model:
        main()

    if save_model:
        model = pickle.load(open(model_path + ".p", "rb"))

        print()
        print("-" * 25)
        print("-" * 5 + "Check Model Save" + "-" * 5)
        print("-" * 25)

        acc_t, mae_t_mean, mae_t_std = model.model_accuracy(
            dataset_type=DatasetType.TRAIN, accuracy_measure=mean_absolute_error
        )
        acc_v, mae_v_mean, mae_v_std = model.model_accuracy(
            dataset_type=DatasetType.VALIDATE, accuracy_measure=mean_absolute_error
        )
        acc_tst, mae_tst_mean, mae_tst_std = model.model_accuracy(
            dataset_type=DatasetType.TEST, accuracy_measure=mean_absolute_error
        )
        print()
        print(f"Train PC NMAE: {acc_t.mean(axis=0)}")
        print(f"Validate PC NMAE: {acc_v.mean(axis=0)}")
        # print(f"Test PC NMAE: {acc_tst.mean(axis=0)}")
        print()
        print(f"Train NMAE: {mae_t_mean:.4e}")
        print(f"Validate NMAE: {mae_v_mean:.4e}")
        # print(f"Test NMAE: {mae_tst_mean:.4e}")
        print()
        print(f"Train STDNAE: {mae_t_std:.4e}")
        print(f"Validate STDNAE: {mae_v_std:.4e}")
        # print(f"Test STDNAE: {mae_tst_std:.4e}")

        train_scores = np.zeros((len(model.datasets.train.input), 5))
        train_scores_pred = np.zeros((len(model.datasets.train.input), 5))

        validate_scores = np.zeros((len(model.datasets.validate.input), 5))
        validate_scores_pred = np.zeros((len(model.datasets.validate.input), 5))

        test_scores = np.zeros((len(model.datasets.test.input), 5))
        test_scores_pred = np.zeros((len(model.datasets.test.input), 5))

        # plot correlations
        rows = 1
        cols = n_pc_scores
        fig, axs = plt.subplots(rows, cols)
        for i in range(n_pc_scores):

            train_y, train_y_pred, _ = model.output_correlation(
                output_index=i, dataset_type=DatasetType.TRAIN
            )
            validate_y, validate_y_pred, _ = model.output_correlation(
                output_index=i, dataset_type=DatasetType.VALIDATE
            )
            test_y, test_y_pred, _ = model.output_correlation(
                output_index=i, dataset_type=DatasetType.TEST
            )

            train_scores_pred[:, i] = train_y_pred
            train_scores[:, i] = train_y
            validate_scores_pred[:, i] = validate_y_pred
            validate_scores[:, i] = validate_y
            test_scores_pred[:, i] = test_y_pred
            test_scores[:, i] = test_y

            col = i % cols
            row = math.floor(i / cols)
            axs[col].set_box_aspect(1)
            axs[col].scatter(
                train_y, train_y_pred, color="red", alpha=0.5, label="Train"
            )
            # axs[col].scatter(validate_y, validate_y_pred, color='blue', alpha=0.5, label="Validate")
            axs[col].scatter(test_y, test_y_pred, color="blue", alpha=0.5, label="Test")
            axs[col].plot([-1, 1], [-1, 1], "k")

            axs[col].set_xlim(left=-1.1, right=1.1)
            axs[col].set_ylim(bottom=-1.1, top=1.1)

            axs[col].set_title("PC" + str(i + 1), fontsize=12)

            if i == 0:
                axs[col].legend()
            if row != rows - 1:
                axs[col].tick_params(bottom=False)
                axs[col].tick_params(labelbottom=False)
            if col != 0:
                axs[col].tick_params(left=False)
                axs[col].tick_params(labelleft=False)

        fig.text(0.5, 0.05, "Truth", ha="center", fontsize=10)
        fig.text(0.05, 0.5, "Prediction", va="center", rotation="vertical", fontsize=10)

        fig.text(
            0.5,
            0.95,
            "Heterogeneous Phase PC Score Parity Plots",
            ha="center",
            fontsize=14,
        )

        # with h5py.File("_Regression/score_predictions_and_truth.hdf5", "a") as f:
        #     f.create_dataset("train_scores_pred", data=train_scores_pred, compression='gzip')
        #     f.create_dataset("train_scores", data=train_scores, compression='gzip')
        #     f.create_dataset("validate_scores_pred", data=validate_scores_pred, compression='gzip')
        #     f.create_dataset("validate_scores", data=validate_scores, compression='gzip')
        #     f.create_dataset("test_scores_pred", data=test_scores_pred, compression='gzip')
        #     f.create_dataset("test_scores", data=test_scores, compression='gzip')

        # plot error histograms
        fig2, axs2 = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True)

        nbins_t = 20
        nbins_v = 10
        nbins_tst = 15

        bin_w = 0.05
        bins = np.arange(0, 0.5 + bin_w, bin_w)

        print(len(acc_t))
        print(acc_t.shape)
        # axs2[0].hist(acc_t[:, 0], bins=bins)
        axs2[0].hist(acc_tst[:, 0], bins=bins)  # , color="tab:orange")
        # axs2[0].hist(acc_v[:, 0], bins=bins)
        axs2[0].set_title("PC1 Error", fontsize=12)

        # axs2[1].hist(acc_t[:, 1], bins=bins)
        axs2[1].hist(acc_tst[:, 1], bins=bins)  # , color="tab:orange")
        # axs2[1].hist(acc_v[:, 1], bins=bins)
        axs2[1].set_title("PC2 Error", fontsize=12)

        # axs2[2].hist(acc_t[:, 2], bins=bins)
        axs2[2].hist(acc_tst[:, 2], bins=bins)  # , color="tab:orange")
        # axs2[2].hist(acc_v[:, 2], bins=bins)
        axs2[2].set_title("PC3 Error", fontsize=12)

        # axs2[3].hist(acc_t[:, 3], bins=bins)
        axs2[3].hist(acc_tst[:, 3], bins=bins)  # , color="tab:orange")
        # axs2[3].hist(acc_v[:, 3], bins=bins)
        axs2[3].set_title("PC4 Error", fontsize=12)

        # axs2[4].hist(acc_t[:, 4], bins=bins)
        axs2[4].hist(acc_tst[:, 4], bins=bins)  # , color="tab:orange")
        # axs2[4].hist(acc_v[:, 4], bins=bins)
        axs2[4].set_title("PC5 Error", fontsize=12)

        axs2[0].set_ylabel("Frequency")
        axs2[2].set_xlabel("Normalized Absolute Error")

    plt.show()
