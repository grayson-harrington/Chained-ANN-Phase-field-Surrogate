import pickle
import h5py
import numpy as np

from MLP_Classifier import MLPClassifier, DatasetType

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def cv_classifier(model_input, model_output, n_folds=1, random_seed=0, classifier_kwargs=None):
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
    for train_indices, test_indices in cv_splits:
        split_index = split_index + 1
        print(f"\nCV Split: {split_index}\n")
        print("Creating Machine Learning Model(s)")

        build_classifier(model_input, model_output, train_indices, test_indices, **classifier_kwargs)


def build_classifier(model_input, model_output, train_indices, test_indices, epochs=50, plot_loss_error=False):
    # create and train model
    nn_model = MLPClassifier(
        model_input=model_input,
        model_output=model_output,
        train_ind=train_indices,
        test_ind=test_indices,
        hidden_shape=(5,),
        batch_size=8,  # smaller batches overfit less
        optimizer_params={"init_lr": 0.001, "decay": 1e-02},
        scheduler_params={"n_epochs": epochs},
    )

    t_loss, t_err, tst_loss, tst_err = nn_model.fit(n_epochs=epochs)

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
        ax.plot(list(range(epochs)), t_loss, label="Training Loss")
        ax.plot(list(range(epochs)), tst_loss, label="Test Loss")
        # ax.set_ylim(bottom=0, top=0.35)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title("Homogeneous/Heterogeneous Classification Loss", fontsize=16)
        ax.legend()

        fig2, ax2 = plt.subplots()
        ax2.plot(list(range(epochs)), t_err, "--", label="Training Error")
        ax2.plot(list(range(epochs)), tst_err, "--", label="Test Error")
        ax2.set_xlabel("Epoch", fontsize=14)
        ax2.set_ylabel("Model Error", fontsize=14)
        ax2.set_title("Homogeneous/Heterogeneous Classification Error", fontsize=16)
        ax2.legend()

        plt.show()

    return nn_model


dataset_path = "_datasets/dataset.hdf5"

train_model = True
n_epochs = 50
cv_folds = 6
rnd_seed = 0
plot_progress = False

save_model = False
model_path = "_Classification/_classification_model"

if __name__ == "__main__":

    with h5py.File(dataset_path, "r") as f:
        print(list(f.keys()))
        parameters = f["parameters"][...][:, 0:18]
        homonohomo = f["n_phases"][...] - 1

    if train_model and not save_model:
        # train model with CV and report metrics. Don't save the models produced
        cv_classifier(parameters, homonohomo, n_folds=cv_folds, random_seed=rnd_seed,
                      classifier_kwargs={"epochs": n_epochs,
                                         "plot_loss_error": plot_progress})
    elif train_model and save_model:
        # train model on random fold and save
        print("TODO")  # TODO?
    elif save_model and not train_model:
        # load model and report metrics
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
