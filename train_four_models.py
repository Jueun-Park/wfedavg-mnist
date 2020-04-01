from torch.utils.data import DataLoader
from learner import Learner
from load_and_split_mnist_dataset import concat_data


NUM_WORKERS = 4

for i in range(0, 8, 2):
    td, vd = concat_data(list(range(10)[i:i+4]))
    tdl = DataLoader(td, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
    vdl = DataLoader(vd, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
    learner = Learner(tdl, vdl, lr=0.005, log_interval=100, tensorboard=True, tensorboard_comment=f"{i}-{i+4}")
    learner.learn(5)
    learner.save(f"model/subenv_{i}-{i+4}")
    