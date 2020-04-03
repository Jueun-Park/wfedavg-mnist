
from gan import GenerativeAdversarialNetwork
from load_and_split_mnist_dataset import concat_data

NUM_WORKERS = 4
epochs = 100


for i in range(0, 8, 2):
    td, vd = concat_data(list(range(10))[i:i+4], mode="tensor")
    gan = GenerativeAdversarialNetwork(save_path=f"./model/subenv_{i}-{i+4}/gan")
    gan.set_data(td, vd)
    gan.learn(2)
    gan.save()
