from torch.utils.data import DataLoader
from module.learner import Learner
from module.load_and_split_mnist_dataset import concat_data


NUM_WORKERS = 4

num_model_5_indices = [8, 9, 0, 1]
td, vd = concat_data(num_model_5_indices)
tdl = DataLoader(td, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
vdl = DataLoader(vd, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
learner = Learner(tdl, vdl, lr=0.005, log_interval=100,
                    tensorboard=True, tensorboard_comment="8-2")
learner.learn(5)
learner.save(f"model/subenv_8-2")
