import pickle

from CDataset import CDatasetsGenerator
from DatasetType import DatasetType

import torch

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np


class MLPClassifier:
    def __init__(
        self,
        model_input,
        model_output,
        train_ind,
        test_ind,
        optimizer_params,
        scheduler_params,
        hidden_shape=(18, 18, 18),
        batch_size=64,
    ):

        self.batch_size = batch_size

        # load in train and validation datasets
        self.datasets = CDatasetsGenerator(model_input, model_output, train_ind, test_ind)
        self.loader_train = self.datasets.make_loader(DatasetType.TRAIN, batch_size)
        self.loader_test = self.datasets.make_loader(DatasetType.TEST, batch_size)

        # create neural net
        n_input = len(self.datasets.train.input[0])
        n_output = 1

        self.net = Net(n_input=n_input, hidden_shape=hidden_shape, n_output=n_output)
        self.n_params = self.net.n_params

        print()
        print(self.net)
        print()
        print("number of parameters: " + str(self.n_params))

        # Adam optimizer for initial learning rate and gradient descent
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=optimizer_params["init_lr"],
            weight_decay=optimizer_params["decay"],
        )

        self.loss_func = torch.nn.BCELoss()

        # might be nice to make div_factor a param
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=50 * optimizer_params["init_lr"],
            div_factor=50,
            three_phase=True,
            total_steps=scheduler_params["n_epochs"],
        )

    def fit(self, n_epochs):
        # train net
        epochs = range(n_epochs)
        t_loss = np.zeros(len(epochs))
        v_loss = np.zeros(len(epochs))

        t_err = np.zeros(len(epochs))
        v_err = np.zeros(len(epochs))

        print()
        print(f"Epoch\tTrain Loss\tValidation Loss")
        print("-" * 40)
        for epoch in epochs:

            self.net.train()
            loss, err = self.run_pass(DatasetType.TRAIN)
            t_loss[epoch] = loss
            t_err[epoch] = err

            self.net.eval()
            loss, err = self.run_pass(DatasetType.TEST)
            v_loss[epoch] = loss
            v_err[epoch] = err

            self.scheduler.step()

            if epoch % 2 == 0:
                print(f"{epoch:4d}\t{t_loss[epoch]:.4f}\t\t{v_loss[epoch]:.4f}")

        return t_loss, t_err, v_loss, v_err

    def run_pass(self, dataset_type):
        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        n_batches = len(loader)
        running_loss = 0
        for ind, batch in enumerate(loader):
            X, y = batch
            y = torch.unsqueeze(y, 1)

            y_pred = self.net(X)  # input x and predict based on x

            if dataset_type is DatasetType.TRAIN:
                loss = self.loss_func(y_pred, y)  # must be (1. nn output, 2. target)
                running_loss += loss.data.numpy() * X.shape[0]

                self.optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # back propagation, compute gradients
                self.optimizer.step()  # apply gradients
            else:
                with torch.no_grad():
                    loss = self.loss_func(y_pred, y)
                    running_loss += loss.data.numpy() * X.shape[0]

        _, mean_acc, _, _ = self.model_accuracy(dataset_type=dataset_type)

        return running_loss / len(loader.dataset), mean_acc

    def predict(self, dataset_type=DatasetType.TRAIN):
        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        y_pred = self.net(loader.dataset.input).detach().numpy()

        # This is for classification purposes, round to nearest integer before accuracy measure
        y_pred = np.round(y_pred)

        return y_pred

    def model_accuracy(
        self,
        dataset_type=DatasetType.TRAIN,
        accuracy_measure=mean_absolute_error,
        print_report=False,
    ):

        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        y = loader.dataset.output.detach().numpy()
        y_pred = self.net(loader.dataset.input).detach().numpy()

        # This is for classification purposes, round to nearest integer before accuracy measure
        y_pred = np.round(y_pred)

        acc = accuracy_measure(y, y_pred, multioutput="raw_values")
        mean = np.mean(acc)
        std = np.std(acc)

        if print_report:
            print()
            print("-" * 25)
            print()
            print(dataset_type)
            print(classification_report(y, y_pred, digits=3))
            print()
            print(confusion_matrix(y, y_pred))

        return acc, mean, std, y_pred

    def output_correlation(self, dataset_type=DatasetType.TRAIN, accuracy_measure=r2_score):

        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        y = loader.dataset.outputc
        y_pred = self.net(loader.dataset.input).detach().numpy()

        return y, y_pred, accuracy_measure(y, y_pred)

    def get_loader(self, dataset_type):
        loader = None
        if dataset_type is DatasetType.TRAIN:
            loader = self.loader_train
        elif dataset_type is DatasetType.TEST:
            loader = self.loader_test
        return loader

    def save_model(self, file_path):
        with open(file_path + ".p", "wb") as f:
            pickle.dump(self, f)


# Neural Net description
class Net(torch.nn.Module):
    def __init__(self, n_input, hidden_shape, n_output):
        super(Net, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(hidden_shape)):
            # make a new FC layer
            lay = torch.nn.Linear(n_input, hidden_shape[i])
            # initialize with a smarter strategy
            torch.nn.init.xavier_normal_(lay.weight)
            # now apply weight normalization
            # add it to our inputs
            self.hidden_layers.append(lay)
            n_input = hidden_shape[i]

        # also make predictor layer and initialize that, then apply weight norm
        self.predict = torch.nn.Linear(n_input, n_output)
        torch.nn.init.xavier_normal_(self.predict.weight)
        self.predict = torch.nn.utils.weight_norm(self.predict)

        # number of parameters in model
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = torch.nn.functional.relu(x)
        x = self.predict(x)
        # sigmoid output to keep between 0-1
        x = torch.sigmoid(x)
        return x
