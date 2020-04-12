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


def model_align(w, base_parameter_dict, sub_model_parameters, alpha=0.5):
    keys = base_parameter_dict.keys()
    for key in keys:
        layer_parameter = []
        for i in range(4):
            layer_parameter.append(sub_model_parameters[i][key].numpy())
        # weighted average
        delta = np.average(layer_parameter, axis=0, weights=w)
        base_parameter_dict[key] = (
            1-alpha) * base_parameter_dict[key] + alpha*delta

# configure
weights = grid_weights_gen()
base_idx = 3
alpha = 0.5
use_cuda = True

if __name__ == "__main__":
    num_model_4_indices = [list(range(10))[i:i+4] for i in range(0, 8, 2)]
    num_model_4_comments = [str(i)+"-"+str(i+4) for i in range(0, 8, 2)]

    base_model = Net()

    base_model.load_state_dict(torch.load(
        f"model/subenv_{num_model_4_comments[base_idx]}/mnist_cnn.pt"))
    
    num_model_4_comments = [str(i)+"-"+str(i+4) for i in range(0, 8, 2)]
    sub_model_parameters = []
    for i in range(4):
        net = Net()
        net.load_state_dict(torch.load(f"./wfed_model_base{base_idx}/subenv_{num_model_4_comments[i]}/mnist_cnn.pt"))
        sub_model_parameters.append(net.state_dict())
        del net

    # load base subenv dataset
    base_train_ds, base_valid_ds = concat_data(
        num_model_4_indices[base_idx], mode="dataset")

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
            print(f"w test in progressing {i}/{len(weights)}")
    
    Path("log").mkdir(parents=True, exist_ok=True)
    with open(f"log/wfedavg_log_base{base_idx}.csv", "w", newline="") as f:
        wf = csv.writer(f)
        wf.writerow(labels)
        wf.writerow(accuracies)
