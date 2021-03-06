import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import csv
from pathlib import Path

from module.load_and_split_mnist_dataset import concat_data
from module.net import Net
from module.learner import Learner
from module.gen_weights import grid_weights_gen
from info import model_comments, model_indices, num_models, grid_size


parser = argparse.ArgumentParser()
parser.add_argument("--base-index", type=int)
args = parser.parse_args()
base_idx = args.base_index

# configure
weights = [[1/num_models for _ in range(num_models)]]
alpha = 0.8
use_cuda = True


def model_align(w, base_parameter_dict, sub_model_parameters, alpha=0.5):
    keys = base_parameter_dict.keys()
    for key in keys:
        layer_parameter = []
        for i in range(num_models):
            layer_parameter.append(sub_model_parameters[i][key].numpy())
        # weighted average
        delta = np.average(layer_parameter, axis=0, weights=w)
        base_parameter_dict[key] = (
            1-alpha) * base_parameter_dict[key] + alpha*delta


if __name__ == "__main__":
    base_model = Net()

    base_model.load_state_dict(torch.load(
        f"model/subenv_{model_comments[base_idx]}/mnist_cnn.pt"))
    
    sub_model_parameters = []
    for i in range(num_models):
        net = Net()
        net.load_state_dict(torch.load(f"./wfed_model_base{base_idx}/subenv_{model_comments[i]}/mnist_cnn.pt"))
        sub_model_parameters.append(net.state_dict())
        del net

    # load base subenv dataset
    base_train_ds, base_valid_ds = concat_data(
        model_indices[base_idx], mode="dataset")

    test_losses = []
    accuracies = []
    labels = []
    aligned_model = Net()
    for i, w in enumerate(weights):
        # client model align
        learner = Learner(DataLoader(base_train_ds, batch_size=64),
                        DataLoader(base_valid_ds, batch_size=64),
                        use_cuda=use_cuda)
        base_parameter_dict = base_model.state_dict()
        model_align(w, base_parameter_dict, sub_model_parameters, alpha=alpha)
        aligned_model.load_state_dict(base_parameter_dict)

        # evaluate fedavg model
        aligned_model.eval()
        learner.model = aligned_model
        test_loss, accuracy = learner._test()
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        labels.append(f"{w}")
        del base_parameter_dict, learner

        if i % 10 == 0:
            print(f"Base index: {base_idx}, w test in progressing {i}/{len(weights)}")
    
    Path("log").mkdir(parents=True, exist_ok=True)
    with open(f"log/simple_fedavg_log_base{base_idx}.csv", "w", newline="") as f:
        wf = csv.writer(f)
        wf.writerow(labels)
        wf.writerow(test_losses)
        wf.writerow(accuracies)
