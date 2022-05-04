import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


# create train and validation datasets to be used by data loaders
class CDatasetsGenerator:
    def __init__(self, train_location, validate_location, test_location):
        # load in training data
        with h5py.File(train_location, 'r') as f:
            print(list(f.keys()))
            t_output = f["n_phases"][...] - 1
            t_params = f["parameters"][...][:, 0:18]

        # load in training data
        with h5py.File(validate_location, 'r') as f:
            v_output = f["n_phases"][...] - 1
            v_params = f["parameters"][...][:, 0:18]

        # load in test data
        with h5py.File(test_location, 'r') as f:
            tst_output = f["n_phases"][...] - 1
            tst_params = f["parameters"][...][:, 0:18]

        # Prepare Data for Training
        # Normalize input based on training mean = 0, std = 1
        self.u_in = np.mean(t_params, axis=0)
        self.s_in = np.std(t_params, axis=0)

        X_train = (t_params - self.u_in) / self.s_in
        X_validate = (v_params - self.u_in) / self.s_in
        X_test = (tst_params - self.u_in) / self.s_in

        y_train = t_output
        y_validate = v_output
        y_test = tst_output

        # create datasets and return
        self.train = SpinodalDataset(X_train, y_train)
        self.validate = SpinodalDataset(X_validate, y_validate)
        self.test = SpinodalDataset(X_test, y_test)

    def apply_input_normalization(self, input_data):
        return (input_data - self.u_in) / self.s_in


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
