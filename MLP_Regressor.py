from RDataset import RDatasetsGenerator
from DatasetType import DatasetType

import torch
import torch.nn.utils as U
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

import pickle


def RMSE(x, y):
    return torch.sqrt(((x - y) ** 2).sum())


class MLPRegressor:
    def __init__(
        self,
        train_location,
        validate_location,
        test_location,
        optimizer_params,
        loss_func_params,
        scheduler_params,
        hidden_shape=(18, 18, 18),
        batch_size=-1,
        dropout_ratio=0.5,
    ):

        self.batch_size = batch_size

        # load in train and validation datasets
        self.datasets = RDatasetsGenerator(
            train_location, validate_location, test_location
        )

        if batch_size == -1:
            batch_size = len(self.datasets.train.input)

        self.loader_train = DataLoader(
            self.datasets.train, batch_size=batch_size, shuffle=True
        )
        self.loader_validate = DataLoader(
            self.datasets.validate, batch_size=batch_size, shuffle=False
        )
        self.loader_test = DataLoader(
            self.datasets.test, batch_size=batch_size, shuffle=False
        )

        # create neural net
        n_input = len(self.datasets.train.input[0])
        n_output = len(self.datasets.train.output[0])

        self.net = Net(
            n_input=n_input,
            hidden_shape=hidden_shape,
            n_output=n_output,
            dropout_ratio=dropout_ratio,
        )
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

        # Huber loss works the best. play with these parameters
        # delta > 0, reduction = {'none', 'mean', 'sum'}
        self.loss_func = torch.nn.HuberLoss(
            reduction=loss_func_params["reduction"], delta=loss_func_params["delta"]
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=100 * optimizer_params["init_lr"],
            div_factor=100,
            three_phase=True,
            pct_start=0.4,
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
            loss, err = self.run_pass(DatasetType.VALIDATE)
            v_loss[epoch] = loss
            v_err[epoch] = err

            self.scheduler.step()

            if epoch % 5 == 0:
                print(f"{epoch:4d}\t{t_loss[epoch]:.4e}\t\t{v_loss[epoch]:.4e}")

        return t_loss, t_err, v_loss, v_err

    def run_pass(self, dataset_type):
        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        n_batches = len(loader)
        running_loss = 0
        for ind, batch in enumerate(loader):
            X, y = batch

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

        _, mean_acc, _ = self.model_accuracy(dataset_type=dataset_type)

        return running_loss / len(loader.dataset), mean_acc

    def model_accuracy(
        self, dataset_type=DatasetType.TRAIN, accuracy_measure=mean_absolute_error
    ):

        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        y = loader.dataset.output.detach().numpy()
        y_pred = self.net(loader.dataset.input).detach().numpy()

        acc = accuracy_measure(y, y_pred, multioutput="raw_values")
        mean = np.mean(acc)
        std = np.std(acc)

        # return acc, mean, std  # TODO make this return value
        return np.absolute(y - y_pred), mean, std

    def output_correlation(
        self, output_index=0, dataset_type=DatasetType.TRAIN, accuracy_measure=r2_score
    ):

        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        y = loader.dataset.output.detach().numpy()[:, output_index]
        y_pred = self.net(loader.dataset.input).detach().numpy()[:, output_index]

        return y, y_pred, accuracy_measure(y, y_pred)

    def get_loader(self, dataset_type):
        loader = None
        if dataset_type is DatasetType.TRAIN:
            loader = self.loader_train
        elif dataset_type is DatasetType.VALIDATE:
            loader = self.loader_validate
        elif dataset_type is DatasetType.TEST:
            loader = self.loader_test
        return loader

    def save_model(self, file_path):
        with open(file_path + ".p", "wb") as f:
            pickle.dump(self, f)


# Neural Net description
class Net(torch.nn.Module):
    def __init__(self, n_input, hidden_shape, n_output, dropout_ratio):
        super(Net, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()

        nchannels = n_input
        for i in range(len(hidden_shape)):
            lay = torch.nn.Linear(nchannels, hidden_shape[i], bias=False)
            torch.nn.init.xavier_normal_(lay.weight)

            self.hidden_layers.append(lay)
            self.batch_norms.append(torch.nn.BatchNorm1d(num_features=hidden_shape[i]))
            self.acts.append(torch.nn.GELU())
            # self.dropouts.append(torch.nn.Dropout(dropout_ratio))
            nchannels = hidden_shape[i]

        self.predict = torch.nn.Linear(nchannels, n_output)
        torch.nn.init.xavier_normal_(self.predict.weight)

        # number of parameters in model
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = self.batch_norms[i](x)
            # x = self.dropouts[i](x)
            x = self.acts[i](x)  # activation function for hidden layers
        x = self.predict(x)  # linear output

        return x
