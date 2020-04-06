import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from module.load_and_split_mnist_dataset import load_and_split_mnist_dataset
from module.net import Net
from module.learner import Learner


def weights_gen(base_index=0):
    for i in range(11):
        base_w = 0.1*i
        weights = [(1-base_w)/3 for _ in range(4)]
        weights[base_index] = base_w
        yield weights


base_index = 0
alpha = 0.5
test_on_full_dataset = False

if __name__ == "__main__":
    td, vd = load_and_split_mnist_dataset()
    num_model_4_comments = [str(i)+"-"+str(i+4) for i in range(0, 8, 2)]
    base_comment = num_model_4_comments[base_index]
    model = Net()
    sub_model_parameters = [model.state_dict() for _ in range(4)]

    keys = model.state_dict().keys()
    base_parameter_dict = sub_model_parameters[base_index]
    for i in range(4):
        tdl = DataLoader(td[i], batch_size=64, shuffle=True, num_workers=4)
        vdl = DataLoader(vd[i], batch_size=64, shuffle=True, num_workers=4)
        learner = Learner(tdl, vdl, lr=0.001, log_interval=100)
        learner.model.load_state_dict(sub_model_parameters[i])
        learner.learn(1)
        sub_model_parameters[i] = learner.model.state_dict()
        del learner
    test_losses = []
    labels = []
    if test_on_full_dataset:
        use_cuda = False
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=64, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
                batch_size=64, shuffle=True, **kwargs)
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
        if test_on_full_dataset:
            learner = Learner(train_loader, test_loader)
        else:
            learner = Learner(DataLoader(
                td[base_index]), DataLoader(vd[base_index]))
        learner.model = model
        test_losses.append(learner._test())
        labels.append(f"{w[base_index]:.2f}")
    # draw graph
    plt.title(
        f"WFedAvg: base model index={base_index}, alpha={alpha}, full dataset test={test_on_full_dataset}")
    plt.bar(labels, test_losses)
    plt.xlabel("base model weight")
    plt.ylabel("test loss")
    plt.show()
