import numpy as np
from torch.utils.data import DataLoader

from ts_benchmark.utils.data_processing import split_before


def train_val_split(train_data, ratio, seq_len):
    if ratio == 1:
        return train_data, None
    border = int(train_data.shape[0] * ratio)
    if seq_len is not None:
        train_data_value, _ = split_before(train_data, border)
        _, valid_data = split_before(train_data, border - seq_len)
        return train_data_value, valid_data
    train_data_value, valid_data = split_before(train_data, border)
    return train_data_value, valid_data


class SegLoader:
    def __init__(self, data, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data = data
        self.test_labels = data

    def __len__(self):
        if self.mode in {"train", "val", "test"}:
            return (self.data.shape[0] - self.win_size) // self.step + 1
        return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (
                np.float32(self.data[index : index + self.win_size]),
                np.float32(self.test_labels[0 : self.win_size]),
            )
        if self.mode == "val":
            return (
                np.float32(self.data[index : index + self.win_size]),
                np.float32(self.test_labels[0 : self.win_size]),
            )
        if self.mode == "test":
            return (
                np.float32(self.data[index : index + self.win_size]),
                np.float32(self.test_labels[index : index + self.win_size]),
            )
        return (
            np.float32(
                self.data[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            ),
            np.float32(
                self.test_labels[
                    index // self.step * self.win_size : index // self.step * self.win_size
                    + self.win_size
                ]
            ),
        )


def anomaly_detection_data_provider(data, batch_size, win_size=100, step=100, mode="train"):
    dataset = SegLoader(data, win_size, 1, mode)
    shuffle = mode in {"train", "val"}
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
