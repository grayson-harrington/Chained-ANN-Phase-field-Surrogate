import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
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
    def __init__(self, train_location, validate_location, test_location):
        # load in training data
        with h5py.File(train_location, "r") as f:
            print(list(f.keys()))
            t_output = f["pc_scores"][...][:, :5]
            t_input = f["parameters"][...][:, 0:18]
            t_homonohomo = f["n_phases"][...]-1

        # load in training data
        with h5py.File(validate_location, "r") as f:
            v_output = f["pc_scores"][...][:, :5]
            v_input = f["parameters"][...][:, 0:18]
            v_homonohomo = f["n_phases"][...]-1

        # load in test data
        with h5py.File(test_location, "r") as f:
            tst_output = f["pc_scores"][...][:, :5]
            tst_input = f["parameters"][...][:, 0:18]
            tst_homonohomo = f["n_phases"][...]-1

        # Prepare Data for Training
        #  remove all single phase samples as this is a heterogeneous model
        t_hetero = np.where(t_homonohomo == 1)[0]
        t_output = t_output[t_hetero]
        t_input = t_input[t_hetero]

        v_hetero = np.where(v_homonohomo == 1)[0]
        v_output = v_output[v_hetero]
        v_input = v_input[v_hetero]

        tst_hetero = np.where(tst_homonohomo == 1)[0]
        tst_output = tst_output[tst_hetero]
        tst_input = tst_input[tst_hetero]

        # Normalize parameters based on training mean = 0, std = 1
        self.u_in = np.mean(t_input, axis=0)
        self.s_in = np.std(t_input, axis=0)

        X_train = (t_input - self.u_in) / self.s_in
        X_validate = (v_input - self.u_in) / self.s_in
        X_test = (tst_input - self.u_in) / self.s_in

        # scale pc_scores from -1 to 1 based on training min and max
        self.min_out = np.min(np.concatenate((t_output, v_output)), axis=0)
        self.max_out = np.max(np.concatenate((t_output, v_output)), axis=0)

        self.min_y_dst = np.tile(-1, len(self.min_out))
        self.max_y_dst = np.tile(1, len(self.max_out))

        y_train = from_range_to_range(t_output, self.min_out, self.max_out, self.min_y_dst, self.max_y_dst)
        y_validate = from_range_to_range(v_output, self.min_out, self.max_out, self.min_y_dst, self.max_y_dst)
        y_test = from_range_to_range(tst_output, self.min_out, self.max_out, self.min_y_dst, self.max_y_dst)

        # y_train = t_output
        # y_validate = v_output
        # y_test = tst_output

        # create datasets and return
        self.train = SpinodalDataset(X_train, y_train)
        self.validate = SpinodalDataset(X_validate, y_validate)
        self.test = SpinodalDataset(X_test, y_test)

    def apply_input_normalization(self, input_data):
        return (input_data - self.u_in) / self.s_in

    def apply_output_normalization(self, output_data):
        return from_range_to_range(output_data, self.min_out, self.max_out, self.min_y_dst, self.max_y_dst)

    def undo_output_normalization(self, output_data):
        return from_range_to_range(output_data, self.min_y_dst, self.max_y_dst, self.min_out, self.max_out)


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
