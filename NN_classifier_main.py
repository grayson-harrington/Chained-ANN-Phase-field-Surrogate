import pickle
import h5py
import numpy as np

from MLP_Classifier import MLPClassifier, DatasetType

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

dataset_path = "_datasets/dataset.hdf5"

n_epochs = 50

train_model = True
cv_folds = 6
random_seed = 0

plot_loss_error = False

save_model = False
model_path = "_Classification/_classification_model"


def main():
    print()
    print("preparing train/test splits")

    with h5py.File(dataset_path, "r") as f:
        print(list(f.keys()))
        parameters = f["parameters"][...][:, 0:18]
        homonohomo = f["n_phases"][...] - 1

    if cv_folds >= 2:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        cv_splits = kf.split(parameters)
    else:
        inds = np.arange(len(parameters))
        np.random.seed(random_seed)
        np.random.shuffle(inds)
        cutoff = int(len(inds) * 0.85)
        cv_splits = [(inds[:cutoff], inds[cutoff:])]

    # loop through cv_splits and build/test model(s)
    split_index = 0
    for train_indices, test_indices in cv_splits:
        split_index = split_index + 1
        print(f"\nCV Split: {split_index}\n")
        print("Creating Machine Learning Model(s)")

        # create and train model
        nn_model = MLPClassifier(
            model_input=parameters,
            model_output=homonohomo,
            train_ind=train_indices,
            test_ind=test_indices,
            hidden_shape=(5,),
            batch_size=8,  # smaller batches overfit less
            optimizer_params={"init_lr": 0.001, "decay": 1e-02},
            scheduler_params={"n_epochs": n_epochs},
        )

        t_loss, t_err, tst_loss, tst_err = nn_model.fit(n_epochs=n_epochs)

        nn_model.model_accuracy(
            dataset_type=DatasetType.TRAIN,
            accuracy_measure=mean_absolute_error,
            print_report=True,
        )

        nn_model.model_accuracy(
            dataset_type=DatasetType.TEST,
            accuracy_measure=mean_absolute_error,
            print_report=True,
        )

        if plot_loss_error:
            # train/validation loss/accuracy charts
            fig, ax = plt.subplots()
            ax.plot(list(range(n_epochs)), t_loss, label="Training Loss")
            ax.plot(list(range(n_epochs)), tst_loss, label="Test Loss")
            # ax.set_ylim(bottom=0, top=0.35)
            ax.set_xlabel("Epoch", fontsize=14)
            ax.set_ylabel("Loss", fontsize=14)
            ax.set_title("Homogeneous/Heterogeneous Classification Loss", fontsize=16)
            ax.legend()

            fig2, ax2 = plt.subplots()
            ax2.plot(list(range(n_epochs)), t_err, "--", label="Training Error")
            ax2.plot(list(range(n_epochs)), tst_err, "--", label="Test Error")
            ax2.set_xlabel("Epoch", fontsize=14)
            ax2.set_ylabel("Model Error", fontsize=14)
            ax2.set_title("Homogeneous/Heterogeneous Classification Error", fontsize=16)
            ax2.legend()

            plt.show()


if __name__ == "__main__":
    if train_model:
        main()

    if save_model:
        model = pickle.load(open(model_path + ".p", "rb"))

        print()
        print("-" * 25)
        print("-" * 5 + "Check Model Save" + "-" * 5)
        print("-" * 25)

        model.model_accuracy(
            dataset_type=DatasetType.TRAIN,
            accuracy_measure=mean_absolute_error,
            print_report=True,
        )

        model.model_accuracy(
            dataset_type=DatasetType.TEST,
            accuracy_measure=mean_absolute_error,
            print_report=True,
        )
