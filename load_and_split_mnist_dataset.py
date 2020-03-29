import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Subset

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
