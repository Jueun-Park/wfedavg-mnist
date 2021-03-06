import torch
from torch.utils.data import DataLoader

from module.learner import Learner
from module.load_and_split_mnist_dataset import load_and_split_mnist_tensor
from module.rnd import RandomNetworkDistillation


epochs = 25

if __name__ == "__main__":
    use_cuda = False
    ts_dict, _, vs_dict, _ = load_and_split_mnist_tensor()

    for i in range(10):
        rnd = RandomNetworkDistillation(log_interval=1000, lr=1e-5, use_cuda=use_cuda, log_tensorboard=True)
        rnd.set_data(train_tensor=ts_dict[i], test_tensor=vs_dict[i])
        rnd.learn(epochs=epochs)
        rnd.save(f"./model/subenv_{i}/rnd")
