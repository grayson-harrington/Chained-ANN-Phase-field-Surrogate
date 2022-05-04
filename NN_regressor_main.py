import math
import pickle
import numpy as np
import h5py

from MLP_Regressor import MLPRegressor, DatasetType

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

n_epochs = 200
n_pc_scores = 5

train_model = False
save_model = True
model_path = "_regression_model"


def main():
    print()
    print("Creating Machine Learning Model")

    NN_model = MLPRegressor(train_location="datasets/PS-Linkage_train.hdf5",
                            validate_location="datasets/PS-Linkage_validate.hdf5",
                            test_location="datasets/PS-Linkage_test.hdf5",

                            hidden_shape=(20, 20),
                            batch_size=128,  # larger batch size tends to give better results
                            dropout_ratio=0.000,

                            optimizer_params={"lr": 0.007,
                                              "decay": 1e-05},

                            loss_func_params={"reduction": 'mean',
                                              "delta": 1},

                            cos_anneal_params={"T_max": n_epochs,
                                               "eta_min": 1e-07})

    t_loss, t_err, v_loss, v_err = NN_model.fit(n_epochs=n_epochs)

    _, mae_t_mean, mae_t_std = NN_model.model_accuracy(dataset_type=DatasetType.TRAIN,
                                                       accuracy_measure=mean_absolute_error)
    _, mae_v_mean, mae_v_std = NN_model.model_accuracy(dataset_type=DatasetType.VALIDATE,
                                                       accuracy_measure=mean_absolute_error)

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
    ax.set_ylim(bottom=0, top=0.05)

    ax2 = ax.twinx()
    ax2.plot(list(range(n_epochs)), 1 - t_err, '--', label="Training Accuracy")
    ax2.plot(list(range(n_epochs)), 1 - v_err, '--', label="Validation Accuracy")
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

        train_y, train_y_pred, r2_train = NN_model.output_correlation(output_index=i, dataset_type=DatasetType.TRAIN)
        validate_y, validate_y_pred, _ = NN_model.output_correlation(output_index=i, dataset_type=DatasetType.VALIDATE)

        col = i % cols
        row = math.floor(i / cols)
        axs[col].set_box_aspect(1)
        axs[col].scatter(train_y, train_y_pred, color='red', alpha=0.5, label="Training")
        axs[col].scatter(validate_y, validate_y_pred, color='blue', alpha=0.5, label="Validation")
        axs[col].plot([-1, 1], [-1, 1], 'k')

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

    fig.text(0.5, 0.05, 'Truth', ha='center', fontsize=10)
    fig.text(0.05, 0.5, 'Prediction', va='center', rotation='vertical', fontsize=10)

    fig.text(0.5, 0.95, "Heterogeneous Phase PC Score Parity Plots", ha='center', fontsize=14)

    # plt.show()


if __name__ == '__main__':
    if train_model:
        main()

    if save_model:
        model = pickle.load(open(model_path + ".p", "rb"))

        print()
        print("-" * 25)
        print("-" * 5 + "Check Model Save" + "-" * 5)
        print("-" * 25)

        _, mae_t_mean, mae_t_std = model.model_accuracy(dataset_type=DatasetType.TRAIN,
                                                        accuracy_measure=mean_absolute_error)
        _, mae_v_mean, mae_v_std = model.model_accuracy(dataset_type=DatasetType.VALIDATE,
                                                        accuracy_measure=mean_absolute_error)

        print()
        print("Train NMAE: %.4f" % mae_t_mean)
        print("Validate NMAE: %.4f" % mae_v_mean)
        print()
        print("Train STDNAE: %.4f" % mae_t_std)
        print("Validate STDNAE: %.4f" % mae_v_std)

        train_scores = np.zeros((len(model.datasets.train.input), 5))
        train_scores_pred = np.zeros((len(model.datasets.train.input), 5))

        # plot correlations
        rows = 1
        cols = n_pc_scores
        fig, axs = plt.subplots(rows, cols)
        for i in range(n_pc_scores):

            train_y, train_y_pred, r2_train = model.output_correlation(output_index=i,
                                                                       dataset_type=DatasetType.TRAIN)
            validate_y, validate_y_pred, _ = model.output_correlation(output_index=i,
                                                                      dataset_type=DatasetType.VALIDATE)

            train_scores_pred[:, i] = train_y_pred
            train_scores[:, i] = train_y

            col = i % cols
            row = math.floor(i / cols)
            axs[col].set_box_aspect(1)
            axs[col].scatter(train_y, train_y_pred, color='red', alpha=0.5, label="Training")
            axs[col].scatter(validate_y, validate_y_pred, color='blue', alpha=0.5, label="Validation")
            axs[col].plot([-1, 1], [-1, 1], 'k')

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

        fig.text(0.5, 0.05, 'Truth', ha='center', fontsize=10)
        fig.text(0.05, 0.5, 'Prediction', va='center', rotation='vertical', fontsize=10)

        fig.text(0.5, 0.95, "Heterogeneous Phase PC Score Parity Plots", ha='center', fontsize=14)

    with h5py.File("PS-Linkage_train_score_predictions.hdf5", "a") as f:
        print(train_scores.shape)
        f.create_dataset("scores_pred", data=train_scores_pred, compression='gzip')
        f.create_dataset("scores", data=train_scores, compression='gzip')

    plt.show()
