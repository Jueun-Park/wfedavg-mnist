from torch.utils.data import DataLoader
from learner import Learner
from load_and_split_mnist_dataset import load_and_split_mnist_dataset

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
