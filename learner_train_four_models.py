from torch.utils.data import DataLoader
from module.learner import Learner
from module.load_and_split_mnist_dataset import concat_data


NUM_WORKERS = 4
num_model_4_indices = [list(range(10))[i:i+4] for i in range(0, 8, 2)]
num_model_4_comments = [str(i)+"-"+str(i+4) for i in range(0, 8, 2)]

for i, index in enumerate(num_model_4_indices):
    td, vd = concat_data(index)
    tdl = DataLoader(td, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
    vdl = DataLoader(vd, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
    learner = Learner(tdl, vdl, lr=0.005, log_interval=100,
                      tensorboard=True, tensorboard_comment=num_model_4_comments[i])
    learner.learn(10)
    learner.save(f"model/subenv_{num_model_4_comments[i]}")
    print(f"save subenv_{num_model_4_comments[i]}")
