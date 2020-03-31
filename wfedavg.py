import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from load_and_split_mnist_dataset import load_and_split_mnist_dataset
from net import Net
from learner import Learner


def weights_gen(base_index=0):
    for i in range(11):
        base_w = 0.1*i
        weights = [(1-base_w)/3 for _ in range(4)]
        weights[base_index] = base_w
        yield weights


def num_model_4_comment_gen():
    for i in range(0, 8, 2):
        yield str(i)+"-"+str(i+4)


base_index = 3
alpha = 0.5

if __name__ == "__main__":

    sub_model_parameters = []

    for idx in num_model_4_comment_gen():
        model = Net()
        model.load_state_dict(torch.load(f"model/subenv_{idx}/checkpoint.pt"))  # checkpoint using early stopping
        # model.load_state_dict(torch.load(f"models/subenv_{idx}/mnist_cnn.pt"))  # trained constant step
        model.eval()
        sub_model_parameters.append(model.state_dict())
        del model
    # print(sub_model_parameters)

    td, vd = load_and_split_mnist_dataset()
    model = Net()
    keys = model.state_dict().keys()
    base_parameter_dict = sub_model_parameters[base_index]
    test_losses = []
    labels = []
    for w in weights_gen(base_index=base_index):
        print(w)
        for key in keys:
            layer_parameter = []
            for i in range(4):
                layer_parameter.append(sub_model_parameters[i][key].numpy())
            # weighted average
            delta = np.average(layer_parameter, axis=0, weights=w)
            base_parameter_dict[key] = (
                1-alpha) * base_parameter_dict[key] + alpha*delta

            # register parameters
        model.load_state_dict(base_parameter_dict)
        # eval weighted model
        model.eval()
        learner = Learner(DataLoader(
            td[base_index]), DataLoader(vd[base_index]))
        learner.model = model
        test_losses.append(learner._test())
        labels.append(f"{w[base_index]:.2f}")
    # draw graph
    plt.title(f"WFedAvg: base model index={base_index}, alpha={alpha}")
    plt.bar(labels, test_losses)
    plt.xlabel("base model weight")
    plt.ylabel("test loss")
    plt.show()
