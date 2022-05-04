import pickle

from CDataset import CDatasetsGenerator
from DatasetType import DatasetType

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np


class MLPClassifier:
    def __init__(self, train_location, validate_location, test_location,
                 optimizer_params,
                 cos_anneal_params,
                 hidden_shape=(18, 18, 18), batch_size=64):

        self.batch_size = batch_size

        # load in train and validation datasets
        self.datasets = CDatasetsGenerator(train_location,
                                           validate_location,
                                           test_location)
        self.loader_train = DataLoader(self.datasets.train, batch_size=batch_size, shuffle=True)
        self.loader_validate = DataLoader(self.datasets.validate, batch_size=batch_size, shuffle=True)
        self.loader_test = DataLoader(self.datasets.test, batch_size=batch_size, shuffle=True)

        # create neural net
        n_input = len(self.datasets.train.input[0])
        n_output = 1

        self.net = Net(n_input=n_input, hidden_shape=hidden_shape, n_output=n_output)
        self.n_params = self.net.n_params

        print()
        print(self.net)
        print()
        print("number of parameters: "+str(self.n_params))

        # Adam optimizer for initial learning rate and gradient descent
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=optimizer_params["lr"],
                                          weight_decay=optimizer_params["decay"],)

        self.loss_func = torch.nn.BCELoss()  # Binary Cross-Entropy Loss

        # cosine annealing learning rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=cos_anneal_params["T_max"],
                                                                    eta_min=cos_anneal_params["eta_min"])

    def fit(self, n_epochs):
        # train net
        epochs = range(n_epochs)
        t_loss = np.zeros(len(epochs))
        v_loss = np.zeros(len(epochs))

        t_err = np.zeros(len(epochs))
        v_err = np.zeros(len(epochs))

        print()
        print(f'Epoch\tTrain Loss\tValidation Loss')
        print('-' * 40)
        for epoch in epochs:

            loss, err = self.run_pass(DatasetType.TRAIN)
            t_loss[epoch] = loss
            t_err[epoch] = err

            loss, err = self.run_pass(DatasetType.VALIDATE)
            v_loss[epoch] = loss
            v_err[epoch] = err

            self.cosine_annealing_lr()

            if epoch % 2 == 0:
                print(f'{epoch:4d}\t{t_loss[epoch]:.4f}\t\t{v_loss[epoch]:.4f}')

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
                running_loss += loss.data.numpy() / n_batches

                self.optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # back propagation, compute gradients
                self.optimizer.step()  # apply gradients
            else:
                with torch.no_grad():
                    loss = self.loss_func(y_pred, y)
                    running_loss += loss.data.numpy() / n_batches

        _, mean_acc, _, _ = self.model_accuracy(dataset_type=dataset_type)

        return running_loss, mean_acc

    def cosine_annealing_lr(self):
        self.scheduler.step()

    def predict(self, dataset_type=DatasetType.TRAIN):
        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        y_pred = self.net(loader.dataset.input).detach().numpy()

        # This is for classification purposes, round to nearest integer before accuracy measure
        y_pred = np.round(y_pred)

        return y_pred

    def model_accuracy(self, dataset_type=DatasetType.TRAIN, accuracy_measure=mean_absolute_error, print_report=False):

        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        y = loader.dataset.output.detach().numpy()
        y_pred = self.net(loader.dataset.input).detach().numpy()

        # This is for classification purposes, round to nearest integer before accuracy measure
        y_pred = np.round(y_pred)
        
        acc = accuracy_measure(y, y_pred, multioutput='raw_values')
        mean = np.mean(acc)
        std = np.std(acc)

        if print_report:
            print()
            print("-"*25)
            print()
            print(dataset_type)
            print(classification_report(y, y_pred))
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
        elif dataset_type is DatasetType.VALIDATE:
            loader = self.loader_validate
        elif dataset_type is DatasetType.TEST:
            loader = self.loader_test
        return loader

    def save_model(self, file_path):
        with open(file_path+".p", "wb") as f:
            pickle.dump(self, f)


# Neural Net description
class Net(torch.nn.Module):
    def __init__(self, n_input, hidden_shape, n_output):
        super(Net, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(hidden_shape)):
            self.hidden_layers.append(torch.nn.Linear(n_input, hidden_shape[i]))
            n_input = hidden_shape[i]
        self.predict = torch.nn.Linear(n_input, n_output)

        # number of parameters in model
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        for i in range(len(self.hidden_layers)):
            x = torch.relu(self.hidden_layers[i](x))  # activation function for hidden layers
        x = torch.sigmoid(self.predict(x))  # sigmoid output to keep between 0-1
        return x
