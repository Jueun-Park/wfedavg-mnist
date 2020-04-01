from rnd import RandomNetworkDistillation
from load_and_split_mnist_dataset import concat_data

NUM_WORKERS = 4

for i in range(0, 8, 2):
    td, vd = concat_data(list(range(10))[i:i+4], mode="tensor")
    rnd = RandomNetworkDistillation(
        log_interval=100, lr=1e-5, use_cuda=False, verbose=1, log_tensorboard=True, path=f"model/subenv_{i}-{i+4}/rnd_model/")
    rnd.set_data(td, vd)
    rnd.learn(10)
    rnd.save()
