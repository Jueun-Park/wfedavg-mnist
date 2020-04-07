import torch
from torch.utils.data import DataLoader

from module.net import Net
from module.learner import Learner
from module.load_and_split_mnist_dataset import concat_data


base_idx = 3
use_cuda = True

if __name__ == "__main__":
    num_model_4_indices = [list(range(10))[i:i+4] for i in range(0, 8, 2)]
    num_model_4_comments = [str(i)+"-"+str(i+4) for i in range(0, 8, 2)]
    # load base model parameter
    base_model = Net()
    base_model.load_state_dict(torch.load(
        f"model/subenv_{num_model_4_comments[base_idx]}/mnist_cnn.pt"))
    sub_model_parameters = [base_model.state_dict() for _ in range(4)]

    # client training
    for i, data_idx in enumerate(num_model_4_indices):
        td, vd = concat_data(data_idx, mode="dataset")
        tdl = DataLoader(td, batch_size=64, shuffle=True)
        vdl = DataLoader(vd, batch_size=64, shuffle=True)
        learner = Learner(tdl, vdl, lr=0.001, log_interval=100, use_cuda=use_cuda)
        learner.model.load_state_dict(sub_model_parameters[i])
        learner.learn(1)
        sub_model_parameters[i] = learner.model.state_dict()
        learner.save(f"./wfed_model_base{base_idx}/subenv_{num_model_4_comments[i]}")
        del learner