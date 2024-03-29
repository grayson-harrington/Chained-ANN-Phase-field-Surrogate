from RDataset import RDatasetsGenerator
from DatasetType import DatasetType

import torch

from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

import pickle


class MLPRegressor:
    def __init__(
            self,
            model_input,
            model_output,
            train_ind,
            test_ind,
            optimizer_params,
            loss_func_params,
            scheduler_params,
            hidden_shape=(18, 18, 18),
            batch_size=-1,
            dropout_ratio=0.5,
    ):

        self.batch_size = batch_size

        # load in train and validation datasets
        self.datasets = RDatasetsGenerator(model_input, model_output, train_ind, test_ind)

        if batch_size == -1:
            batch_size = len(self.datasets.train.input)

        self.loader_train = self.datasets.make_loader(DatasetType.TRAIN, batch_size)
        self.loader_test = self.datasets.make_loader(DatasetType.TEST, batch_size)

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
        self.loss_func = torch.nn.HuberLoss(reduction=loss_func_params["reduction"], delta=loss_func_params["delta"])
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
        tst_loss = np.zeros(len(epochs))

        t_err = np.zeros(len(epochs))
        tst_err = np.zeros(len(epochs))

        print()
        print(f"Epoch\tTrain Loss\tValidation Loss")
        print("-" * 40)
        for epoch in epochs:

            self.net.train()
            loss, err = self.run_pass(DatasetType.TRAIN)
            t_loss[epoch] = loss
            t_err[epoch] = np.mean(err)

            self.net.eval()
            loss, err = self.run_pass(DatasetType.TEST)
            tst_loss[epoch] = loss
            tst_err[epoch] = np.mean(err)

            self.scheduler.step()

            if epoch % 5 == 0:
                print(f"{epoch:4d}\t{t_loss[epoch]:.4e}\t\t{tst_loss[epoch]:.4e}")

        return t_loss, t_err, tst_loss, tst_err

    def run_pass(self, dataset_type):
        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

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

        _, _, mean_acc, _, _ = self.model_accuracy(dataset_type=dataset_type)

        return running_loss / len(loader.dataset), mean_acc

    # Calculate nMAE and std for each PC score
    def nMAE_nSTD_r2(self, truth, pred, print_metrics=False):
        # determine the normalization factors
        norm_denom = np.mean(np.abs(truth), axis=0)

        # calculate normalized absolute error for each value
        nae = np.abs(truth - pred) / norm_denom

        # calculate nMAE
        nmae = np.mean(nae, axis=0)
        nstd = np.std(nae, axis=0)

        # r2
        r2 = [r2_score(truth[:, i], pred[:, i]) for i in range(5)]

        if print_metrics:
            print("nmae:\t" + str(nmae))
            print("nstd:\t" + str(nstd))
            print("r2:\t" + str(r2))

        return nae, nmae, nstd, r2

    def predict(self, model_input, scale_input=False, unscale_output=False):

        if scale_input:
            model_input = self.datasets.apply_input_normalization(model_input)

        if isinstance(model_input, np.ndarray):
            model_input = torch.from_numpy(model_input).float()

        # make prediction
        y_pred = self.net(model_input).detach().numpy()
        
        if unscale_output:
            y_pred = self.datasets.undo_output_normalization(y_pred)

        return y_pred

    def model_accuracy(self, dataset_type=DatasetType.TRAIN, print_report=False, unscale_output=False):

        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        y = loader.dataset.output.detach().numpy()
        y_pred = self.net(loader.dataset.input).detach().numpy()

        if unscale_output:
            y = self.datasets.undo_output_normalization(y)
            y_pred = self.datasets.undo_output_normalization(y_pred)

        if print_report:
            print(f"\n{dataset_type}")

        nae, nmae, nstd, r2 = self.nMAE_nSTD_r2(y, y_pred, print_metrics=print_report)

        # return acc, mean, std
        return y, y_pred, nmae, nstd, r2

    def output_correlation(self, output_index=0, dataset_type=DatasetType.TRAIN, accuracy_measure=r2_score,
                           unscale_output=True):

        loader = self.get_loader(dataset_type)
        if loader is None:
            return -1

        y = loader.dataset.output.detach().numpy()
        y_pred = self.net(loader.dataset.input).detach().numpy()

        if unscale_output:
            y = self.datasets.undo_output_normalization(y)[:, output_index]
            y_pred = self.datasets.undo_output_normalization(y_pred)[:, output_index]

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
