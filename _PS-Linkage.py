import math
import pickle
import h5py
import numpy as np

from DatasetType import DatasetType

import torch
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix

import matplotlib.pyplot as plt

classification_file_path = "_Classification/_classification_model"
regression_file_path = "_Regression/_regression_model"

train_location = "datasets/train.hdf5"
validate_location = "datasets/validate.hdf5"
test_location = "datasets/test.hdf5"

dataset = DatasetType.TEST
dataset_compare = DatasetType.TRAIN
n_scores = 5


def main():
    # load in train data
    with h5py.File(train_location, 'r') as f:
        train_classes = f["n_phases"][...] - 1
        train_params = f["parameters"][...][:, 0:18]
        train_scores = f["pc_scores"][...][:, :5]
        train_corrs = f["correlations"][...]
        train_micros = f["curated_micros"][...]

    # load in validation data
    with h5py.File(validate_location, 'r') as f:
        validate_classes = f["n_phases"][...] - 1
        validate_params = f["parameters"][...][:, 0:18]
        validate_scores = f["pc_scores"][...][:, :5]
        validate_corrs = f["correlations"][...]
        validate_micros = f["curated_micros"][...]

    # load in test data
    with h5py.File(test_location, 'r') as f:
        test_classes = f["n_phases"][...] - 1
        test_params = f["parameters"][...][:, 0:18]
        test_scores = f["pc_scores"][...][:, :5]
        test_corrs = f["correlations"][...]
        test_micros = f["curated_micros"][...]

    # get dataset values of interest
    if dataset is DatasetType.TRAIN:
        classes = train_classes
        params = train_params
        scores = train_scores
        corrs = train_corrs
        micros = train_micros
    elif dataset is DatasetType.VALIDATE:
        classes = validate_classes
        params = validate_params
        scores = validate_scores
        corrs = validate_corrs
        micros = validate_micros
    else:
        classes = test_classes
        params = test_params
        scores = test_scores
        corrs = test_corrs
        micros = test_micros

    # load in the classification and regression models
    classification_model = pickle.load(open(classification_file_path + ".p", "rb"))
    regression_model = pickle.load(open(regression_file_path + ".p", "rb"))

    #
    #
    # perform homogeneous/heterogeneous classification
    #
    #

    # apply normalization to process parameters
    params_norm = classification_model.datasets.apply_input_normalization(params)
    params_norm = torch.from_numpy(params_norm).float()

    # get classification predictions and binarize the continuous output
    classes_pred = classification_model.net(params_norm).detach().numpy()
    classes_pred = np.round(classes_pred)

    # print classification report and confusion matrix (validation dataset accuracy confirmed)
    print()
    print(dataset)
    print(classification_report(classes, classes_pred))
    print()
    print(confusion_matrix(classes, classes_pred))
    print()

    #
    #
    # perform pc score prediction on "heterogeneous" samples
    #
    #

    # Retrieve the heterogeneous sample process parameters and pc scores based on the classification model output
    indices = np.where(classes_pred == 1)[0]
    hetero_params = params[indices]
    hetero_scores = scores[indices]
    hetero_corrs = corrs[indices]
    hetero_micros = micros[indices]

    # apply normalization to process parameters
    hetero_params_norm = regression_model.datasets.apply_input_normalization(hetero_params)
    hetero_params_norm = torch.from_numpy(hetero_params_norm).float()

    # apply normalization to true pc_scores
    hetero_scores_norm = regression_model.datasets.apply_output_normalization(hetero_scores)

    # get PC score predictions
    hetero_scores_pred = regression_model.net(hetero_params_norm).detach().numpy()

    # Report model accuracy (validation dataset accuracy confirmed)
    acc = mean_absolute_error(hetero_scores_norm, hetero_scores_pred, multioutput='raw_values')
    mean = np.mean(acc)
    std = np.std(acc)

    print()
    print("PC score prediction accuracy WITH classification error")
    print("NMAE: %.4f" % mean)
    print("STDNAE: %.4f" % std)

    # just checking validation set on regression to make sure that all is running as it should be.
    # run regression on new samples and display accuracy with parity plots and NMAE measures
    _, mae_v_mean, mae_v_std = regression_model.model_accuracy(dataset_type=dataset,
                                                               accuracy_measure=mean_absolute_error)
    print()
    print("PC score prediction accuracy WITHOUT classification error")
    print("NMAE: %.4f" % mae_v_mean)
    print("STDNAE: %.4f" % mae_v_std)

    with h5py.File("_Regression/score_predictions_and_truth.hdf5", "a") as f:
        train_scores_pred = f['train_scores_pred'][...]
        train_scores_true = f['train_scores'][...]

        print(train_scores_pred.shape)
        print(train_scores_true.shape)

    # plot correlations
    rows = 1
    cols = 5
    fig, axs = plt.subplots(rows, cols)
    for i in range(rows * cols):

        col = i % cols
        row = math.floor(i / cols)
        axs[col].set_box_aspect(1)
        axs[col].scatter(train_scores_true[:, i], train_scores_pred[:, i],
                         color='red', alpha=0.5, label="Train")
        axs[col].scatter(hetero_scores_norm[:, i], hetero_scores_pred[:, i],
                         color='blue', alpha=0.5, label="Test")
        axs[col].plot([-1, 1], [-1, 1], 'k')

        axs[col].set_xlim(left=-1.1, right=1.1)
        axs[col].set_ylim(bottom=-1.1, top=1.1)

        axs[col].set_title("PC" + str(i + 1), fontsize=12)

        if i == 0:
            axs[col].legend()
        if col != 0:
            axs[col].tick_params(left=False)
            axs[col].tick_params(labelleft=False)

    fig.text(0.5, 0.05, 'Truth', ha='center', fontsize=12)
    fig.text(0.05, 0.5, 'Prediction', va='center', rotation='vertical', fontsize=12)
    fig.text(0.5, 0.95, "PC Score Prediction for Heterogeneous Classified Samples", ha='center', fontsize=14)

    # unscale the PC score predictions and compare with truth
    hetero_scores_true = hetero_scores
    hetero_scores_pred = regression_model.datasets.undo_output_normalization(hetero_scores_pred)

    # unscale the PC score TRAIN predictions and compare with truth
    hetero_scores_train_true = regression_model.datasets.undo_output_normalization(train_scores_true)
    hetero_scores_train_pred = regression_model.datasets.undo_output_normalization(train_scores_pred)

    # plot correlations
    rows = 1
    cols = 5
    fig2, axs2 = plt.subplots(rows, cols)
    for i in range(rows * cols):

        col = i % cols
        row = math.floor(i / cols)
        axs2[col].set_box_aspect(1)
        axs2[col].scatter(hetero_scores_train_true[:, i], hetero_scores_train_pred[:, i], color='red', alpha=0.5, label="Train")
        axs2[col].scatter(hetero_scores_true[:, i], hetero_scores_pred[:, i], color='blue', alpha=0.5, label="Test")

        mins1 = np.min(np.concatenate((hetero_scores_true[:, i], hetero_scores_pred[:, i])), axis=None)
        maxs1 = np.max(np.concatenate((hetero_scores_true[:, i], hetero_scores_pred[:, i])), axis=None)
        mins2 = np.min(np.concatenate((hetero_scores_train_true[:, i], hetero_scores_train_pred[:, i])), axis=None)
        maxs2 = np.max(np.concatenate((hetero_scores_train_true[:, i], hetero_scores_train_pred[:, i])), axis=None)

        mins = min(mins1, mins2)
        maxs = max(maxs1, maxs2)

        axs2[col].plot([mins, maxs], [mins, maxs], 'k')

        axs2[col].set_xlim(left=mins+mins/10, right=maxs+maxs/10)
        axs2[col].set_ylim(bottom=mins+mins/10, top=maxs+maxs/10)

        axs2[col].set_title("PC" + str(i + 1), fontsize=12)

        if i == 0:
            axs2[col].legend()
        if col != 0:
            axs2[col].tick_params(left=False)
            axs2[col].tick_params(labelleft=False)

    fig2.text(0.5, 0.05, 'Truth', ha='center', fontsize=12)
    fig2.text(0.05, 0.5, 'Prediction', va='center', rotation='vertical', fontsize=12)
    fig2.text(0.5, 0.95, "PC Score Prediction for Heterogeneous Classified Samples", ha='center', fontsize=14)

    # plt.show()

    # save file with predicted scores, true scores, parameters, correlations, micros
    with h5py.File("_PS Linkage/PS-Linkage_RESULTS_final.hdf5", "a") as f:
        f.create_dataset("pc_scores_true", data=hetero_scores_true, compression='gzip')
        f.create_dataset("pc_scores_pred", data=hetero_scores_pred, compression='gzip')
        f.create_dataset("parameters", data=hetero_params, compression='gzip')
        f.create_dataset("correlations_true", data=hetero_corrs, compression='gzip')
        f.create_dataset("curated_micros_true", data=hetero_micros, compression='gzip')


if __name__ == '__main__':
    main()

    plt.show()
