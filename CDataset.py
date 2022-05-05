import h5py
import numpy as np

from DatasetType import DatasetType

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


# create train and validation datasets to be used by data loaders
class CDatasetsGenerator:
    def __init__(self, data_input, data_output, train_indices, test_indices):
        # get training data
        t_input = data_input[train_indices]
        t_output = data_output[train_indices]

        # get test data
        tst_input = data_input[test_indices]
        tst_output = data_output[test_indices]

        # Prepare Data for Training
        # Normalize input based on training mean = 0, std = 1
        self.u_in = np.mean(t_input, axis=0)
        self.s_in = np.std(t_input, axis=0)

        X_train = self.apply_input_normalization(t_input)
        X_test = self.apply_input_normalization(tst_input)

        y_train = t_output
        y_test = tst_output

        # create datasets
        self.train = SpinodalDataset(X_train, y_train)
        self.test = SpinodalDataset(X_test, y_test)

    def apply_input_normalization(self, input_data):
        return (input_data - self.u_in) / self.s_in

    def make_loader(self, dataset_type, batch_size):
        dataset = self.train if dataset_type is DatasetType.TRAIN else self.test
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Dataset
class SpinodalDataset(Dataset):
    def __init__(self, inputs, outputs):
        # Convert data to torch.Tensor
        self.input = torch.from_numpy(inputs).float()
        self.output = torch.from_numpy(outputs).float()

    def __getitem__(self, index) -> T_co:
        return self.input[index], self.output[index]

    def __len__(self):
        return len(self.input)
