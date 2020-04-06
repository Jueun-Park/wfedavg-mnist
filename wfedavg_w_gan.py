import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

from module.load_and_split_mnist_dataset import concat_data
from module.net import Net
from module.learner import Learner
from module.gan import GenerativeAdversarialNetwork


def weights_gen(base_index=0):
    for i in range(11):
        base_w = 0.1*i
        weights = [(1-base_w)/3 for _ in range(4)]
        weights[base_index] = base_w
        yield weights


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


alpha = 0.5
base_idx = 1

if __name__ == "__main__":
    num_model_4_indices = [list(range(10))[i:i+4] for i in range(0, 8, 2)]
    num_model_4_comments = [str(i)+"-"+str(i+4) for i in range(0, 8, 2)]

    # load base subenv dataset
    base_train_ts, base_valid_ts = concat_data(
        num_model_4_indices[base_idx], mode="tensor")

    # get weight from base gan model
    gan_weights = []
    for data_com in num_model_4_comments:
        gan = GenerativeAdversarialNetwork(
            save_path=f"./model/subenv_{data_com}/gan")
        gan.load()
        w = gan.get_discriminator_output(base_valid_ts)
        gan_weights.append(w.item())
    print(gan_weights)
    gan_weights = softmax(gan_weights)
    print(gan_weights)

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
        learner = Learner(tdl, vdl, lr=0.001, log_interval=100)
        learner.model.load_state_dict(sub_model_parameters[i])
        learner.learn(1)
        sub_model_parameters[i] = learner.model.state_dict()
        del learner

    # load base subenv dataset
    base_train_ds, base_valid_ds = concat_data(
        num_model_4_indices[base_idx], mode="dataset")

    test_losses = []
    labels = []
    aligned_model = Net()
    for w in weights_gen(base_index=base_idx):
        # client model align
        learner = Learner(DataLoader(base_train_ds, batch_size=64),
                        DataLoader(base_valid_ds, batch_size=64))
        base_parameter_dict = base_model.state_dict()
        model_align(w, base_parameter_dict, sub_model_parameters, alpha=alpha)
        aligned_model.load_state_dict(base_parameter_dict)

        # evaluate fedavg model
        aligned_model.eval()
        learner.model = aligned_model
        test_losses.append(learner._test())
        labels.append(f"{w[base_idx]:.2f}")
        del base_parameter_dict, learner

    # eval gan weights
    learner = Learner(DataLoader(base_train_ds, batch_size=64),
                    DataLoader(base_valid_ds, batch_size=64))
    base_parameter_dict = base_model.state_dict()
    model_align(gan_weights, base_parameter_dict,
                sub_model_parameters, alpha=alpha)
    aligned_model.load_state_dict(base_parameter_dict)

    aligned_model.eval()
    learner.model = aligned_model
    test_losses.append(learner._test())
    labels.append("ganw")

    gan_w_print = [float(f"{i:.2f}") for i in gan_weights]
    plt.title(
        f"WFedAvg: base model index={base_idx}, alpha={alpha}, ganw={gan_w_print}")
    plt.bar(labels, test_losses)
    for a, b in zip(labels, test_losses):
        plt.text(a, b, str(f"{b:.3f}"))
    plt.xlabel("base model weight")
    plt.ylabel("test loss")
    plt.show()
