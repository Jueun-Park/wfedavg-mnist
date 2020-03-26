import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader


def array_to_dataloader(x, y):
    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset)


def load_and_split_data():
    train_set = datasets.MNIST('./data', train=True, download=True)
    test_set = datasets.MNIST('./data', train=False, download=True)

    x_train = train_set.data.numpy()
    y_train = train_set.targets.numpy()

    x_valid = test_set.data.numpy()
    y_valid = test_set.targets.numpy()

    train_x_arrs = {}
    train_y_arrs = {}
    for i in range(10):
        train_x_arrs[i] = x_train[y_train == i]
        train_y_arrs[i] = np.full(shape=(x_train.shape[0]), fill_value=i)

    valid_x_arrs = {}
    valid_y_arrs = {}
    for i in range(10):
        valid_x_arrs[i] = x_valid[y_valid == i]
        valid_y_arrs[i] = np.full(shape=(x_valid.shape[0]), fill_value=i)

    return train_x_arrs, train_y_arrs, valid_x_arrs, valid_y_arrs


if __name__ == "__main__":
    txa, tya, vxa, vya = load_and_split_data()
    for i in range(10):
        print(txa[i].shape, tya[i].shape, vxa[i].shape, vya[i].shape)
        # print(array_to_dataloader(txa[i], tya[i]))
