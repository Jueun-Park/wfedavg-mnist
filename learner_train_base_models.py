from torch.utils.data import DataLoader
from module.learner import Learner
from module.load_and_split_mnist_dataset import concat_data
from info import model_comments, model_indices, base_learn_epochs

NUM_WORKERS = 4

for i, index in enumerate(model_indices):
    td, vd = concat_data(index)
    tdl = DataLoader(td, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
    vdl = DataLoader(vd, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
    learner = Learner(tdl, vdl, lr=0.005, log_interval=100,
                      tensorboard=True, tensorboard_comment=model_comments[i])
    learner.learn(base_learn_epochs)
    learner.save(f"model/subenv_{model_comments[i]}")
    print(f"save subenv_{model_comments[i]}")
    del learner
