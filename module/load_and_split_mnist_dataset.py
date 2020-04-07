import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset

from os import path
import sys
sys.path.append(path.abspath(path.dirname(__file__)))
from learner import Learner


def load_mnist():
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_set = datasets.MNIST('./data', train=False, download=True,  transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    return train_set, test_set


def load_and_split_mnist_dataset():
    train_set, test_set = load_mnist()

    x_train = train_set.data.numpy()
    y_train = train_set.targets.numpy()

    x_valid = test_set.data.numpy()
    y_valid = test_set.targets.numpy()

    train_dataset = {}
    for i in range(10):
        train_dataset[i] = Subset(train_set, *np.where(y_train == i))

    valid_dataset = {}
    for i in range(10):
        valid_dataset[i] = Subset(test_set, *np.where(y_valid == i))

    return train_dataset, valid_dataset


def load_and_split_mnist_tensor():
    train_set, test_set = load_mnist()

    x_train = train_set.data.numpy()
    y_train = train_set.targets.numpy()

    x_valid = test_set.data.numpy()
    y_valid = test_set.targets.numpy()

    train_x_tensors = {}
    train_y_tensors = {}

    x_train = x_train[:, np.newaxis, :]  # add channel dim
    x_valid = x_valid[:, np.newaxis, :]  # add channel dim
    for i in range(10):
        train_x_tensors[i] = torch.Tensor(x_train[y_train == i])
        train_y_tensors[i] = np.full(shape=(x_train.shape[0]), fill_value=i)

    valid_x_tensors = {}
    valid_y_tensors = {}
    for i in range(10):
        valid_x_tensors[i] = torch.Tensor(x_valid[y_valid == i])
        valid_y_tensors[i] = np.full(shape=(x_valid.shape[0]), fill_value=i)

    return train_x_tensors, train_y_tensors, valid_x_tensors, valid_y_tensors


def concat_data(index_list, mode="dataset"):
    """make concatenated data from sliced data
    :param index_list: indices you want to concat
    :param mode: return mode "dataset" or "tensor"
    :return: concatenated dataset or tensor
    """
    if mode == "dataset":
        td, vd = load_and_split_mnist_dataset()
        result_t = td[index_list[0]]
        result_v = td[index_list[0]]
        index_list = index_list[1:]
        for i in index_list:
            result_t += td[i]
            result_v += vd[i]
    elif mode == "tensor":
        tx, _, vx, _ = load_and_split_mnist_tensor()
        result_t = tx[index_list[0]]
        result_v = vx[index_list[0]]
        index_list = index_list[1:]
        for i in index_list:
            torch.cat((result_t, tx[i]), dim=0)
            torch.cat((result_v, vx[i]), dim=0)
    else:
        return None
    return result_t, result_v


if __name__ == "__main__":
    import time
    td, vd = concat_data([1, 2, 3, 4])
    learner = Learner(DataLoader(td, batch_size=64, shuffle=True, num_workers=4), DataLoader(
        vd, batch_size=64, shuffle=True, num_workers=4), log_interval=100, lr=0.005)

    st = time.time()
    learner.learn(2)
    print(f"total time: {time.time() - st}")

    td, vd = concat_data([3, 4, 5, 6])
    learner.test_loader = DataLoader(
        vd, batch_size=64, shuffle=True, num_workers=4)
    learner._test()
