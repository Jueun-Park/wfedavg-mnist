import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Subset

from learner import Learner


def load_and_split_mnist_dataset():
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_set = datasets.MNIST('./data', train=False, download=True,  transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    x_train = train_set.data.numpy()
    y_train = train_set.targets.numpy()

    x_valid = test_set.data.numpy()
    y_valid = test_set.targets.numpy()

    train_dataset = {}
    for i in range(10):
        train_dataset[i] = Subset(train_set, *np.where(y_train == i))
        print(np.where(y_train == i)[0].shape)

    valid_dataset = {}
    for i in range(10):
        valid_dataset[i] = Subset(test_set, *np.where(y_valid == i))
        print(np.where(y_valid == i)[0].shape)

    return train_dataset, valid_dataset


if __name__ == "__main__":
    use_cuda = False
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    td, vd = load_and_split_mnist_dataset()
    train_dataloaders = {}
    valid_dataloaders = {}
    for i in range(10):
        train_dataloaders[i] = DataLoader(
            td[i], batch_size=64, shuffle=True, **kwargs)
        valid_dataloaders[i] = DataLoader(
            vd[i], batch_size=64, shuffle=True, **kwargs)

    for i in range(10):
        learner = Learner(train_dataloaders[i], valid_dataloaders[i])
        learner.learn(10)
        learner.save(f"./model/subenv_{i}")
