import argparse
import torch
from torch.utils.data import DataLoader

from module.net import Net
from module.learner import Learner
from module.load_and_split_mnist_dataset import concat_data
from info import model_indices, model_comments, num_models, base_learn_times

parser = argparse.ArgumentParser()
parser.add_argument("--base-index", type=int)
args = parser.parse_args()
base_idx = args.base_index

use_cuda = True

if __name__ == "__main__":
    print(f"Base index: {base_idx}")
    # load base model parameter
    base_model = Net()
    base_model.load_state_dict(torch.load(
        f"model/subenv_{model_comments[base_idx]}/mnist_cnn.pt"))
    sub_model_parameters = [base_model.state_dict() for _ in range(num_models)]

    # client training
    for i, data_idx in enumerate(model_indices):
        print(f"data index: {data_idx}")
        td, vd = concat_data(data_idx, mode="dataset")
        tdl = DataLoader(td, batch_size=64, shuffle=True)
        vdl = DataLoader(vd, batch_size=64, shuffle=True)
        learner = Learner(tdl, vdl, lr=0.001, log_interval=100, use_cuda=use_cuda)
        learner.model.load_state_dict(sub_model_parameters[i])
        learner.learn(base_learn_times)
        sub_model_parameters[i] = learner.model.state_dict()
        learner.save(f"./wfed_model_base{base_idx}/subenv_{model_comments[i]}")
        del learner
