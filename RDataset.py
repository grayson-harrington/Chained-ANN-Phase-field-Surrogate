import numpy as np

from DatasetType import DatasetType

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


# Normalize output based on range mapping to -1:1
def from_range_to_range(mat, src_min, src_max, dst_min, dst_max):
    """
    Scale the given value from the scale of src to the scale of dst.
        src and dst are arrays that match cols of val
        x' = ((x - a) / (b - a)) * (b' - a') + a'
    """

    if np.array_equal(src_min, src_max):
        return np.ones(mat.shape) * dst_max

    src_min = np.tile(src_min, (len(mat), 1))
    src_max = np.tile(src_max, (len(mat), 1))

    dst_min = np.tile(dst_min, (len(mat), 1))
    dst_max = np.tile(dst_max, (len(mat), 1))

    return ((mat - src_min) / (src_max - src_min)) * (dst_max - dst_min) + dst_min


# create train and validation datasets to be used by data loaders
class RDatasetsGenerator:
    def __init__(self, data_input, data_output, train_indices, test_indices):

        # load in training data
        t_input = data_input[train_indices]
        t_output = data_output[train_indices]

        # get test data
        tst_input = data_input[test_indices]
        tst_output = data_output[test_indices]

        # Prepare Data for Training
        # Normalize parameters based on training mean = 0, std = 1
        self.u_in = np.mean(t_input, axis=0)
        self.s_in = np.std(t_input, axis=0)

        X_train = self.apply_input_normalization(t_input)
        X_test = self.apply_input_normalization(tst_input)

        # scale pc_scores from -1 to 1 based on training min and max
        self.min_out = np.min(np.concatenate((t_output, tst_output)), axis=0)
        self.max_out = np.max(np.concatenate((t_output, tst_output)), axis=0)

        self.min_y_dst = np.tile(-1, len(self.min_out))
        self.max_y_dst = np.tile(1, len(self.max_out))

        y_train = self.apply_output_normalization(t_output)
        y_test = self.apply_output_normalization(tst_output)

        # create datasets
        self.train = SpinodalDataset(X_train, y_train)
        self.test = SpinodalDataset(X_test, y_test)

    def apply_input_normalization(self, input_data):
        return (input_data - self.u_in) / self.s_in

    def apply_output_normalization(self, output_data):
        return from_range_to_range(output_data, self.min_out, self.max_out, self.min_y_dst, self.max_y_dst)

    def undo_output_normalization(self, output_data):
        return from_range_to_range(output_data, self.min_y_dst, self.max_y_dst, self.min_out, self.max_out)

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
