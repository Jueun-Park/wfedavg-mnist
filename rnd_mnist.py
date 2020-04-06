from module.rnd import RandomNetworkDistillation
from module.load_and_split_mnist_dataset import concat_data


if __name__ == "__main__":
    rnd = RandomNetworkDistillation(log_interval=1000, lr=1e-3, path="model/allenv/rnd_model/", verbose=1)
    train_tensor, valid_tensor = concat_data(list(range(0, 10)), mode="tensor")
    rnd.set_data(train_tensor, valid_tensor)
    rnd.learn(30)
    rnd.save()
