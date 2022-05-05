import pickle

from MLP_Classifier import MLPClassifier, DatasetType

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

n_epochs = 50

train_model = False
save_model = True
model_path = "_Classification/_classification_model"


def main():
    print()
    print("Creating Machine Learning Model")

    nn_model = MLPClassifier(
        train_location="../6. Early Train Split/datasets/train.hdf5",
        validate_location="../6. Early Train Split/datasets/validate.hdf5",
        test_location="../6. Early Train Split/datasets/test.hdf5",
        hidden_shape=(5,),
        batch_size=8,  # smaller batches overfit less
        optimizer_params={"init_lr": 0.001, "decay": 1e-02},
        scheduler_params={"n_epochs": n_epochs},
    )

    t_loss, t_err, v_loss, v_err = nn_model.fit(n_epochs=n_epochs)

    nn_model.model_accuracy(
        dataset_type=DatasetType.TRAIN,
        accuracy_measure=mean_absolute_error,
        print_report=True,
    )

    nn_model.model_accuracy(
        dataset_type=DatasetType.VALIDATE,
        accuracy_measure=mean_absolute_error,
        print_report=True,
    )

    if save_model:
        nn_model.save_model(model_path)

    # train/validation loss/accuracy charts
    fig, ax = plt.subplots()
    ax.plot(list(range(n_epochs)), t_loss, label="Training Loss")
    ax.plot(list(range(n_epochs)), v_loss, label="Validation Loss")
    # ax.set_ylim(bottom=0, top=0.35)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title("Homogeneous/Heterogeneous Classification Loss", fontsize=16)
    ax.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(list(range(n_epochs)), t_err, "--", label="Training Error")
    ax2.plot(list(range(n_epochs)), v_err, "--", label="Validation Error")
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
            dataset_type=DatasetType.VALIDATE,
            accuracy_measure=mean_absolute_error,
            print_report=True,
        )

        model.model_accuracy(
            dataset_type=DatasetType.TEST,
            accuracy_measure=mean_absolute_error,
            print_report=True,
        )
