import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from module.load_and_split_mnist_dataset import concat_data
from module.learner import Learner
from module.net import Net


use_cuda = True

if __name__ == "__main__":
    num_model_5_indices = [list(range(10))[i:i+4] for i in range(0, 8, 2)] + [[8, 9, 0, 1]]
    num_model_4_indices = [list(range(10))[i:i+4] for i in range(0, 8, 2)]
    num_model_5_comments = [str(i)+"-"+str(i+4) for i in range(0, 8, 2)] + ["8-2"]
    num_model_4_comments = [str(i)+"-"+str(i+4) for i in range(0, 8, 2)]

    Path("log").mkdir(parents=True, exist_ok=True)
    file = open("log/simple_agent_test.csv", "w", newline="")
    writer = csv.writer(file)
    writer.writerow(["sub_data"] + num_model_4_comments)
    for model_name in num_model_4_comments:
        print(model_name)
        model = Net()
        model.load_state_dict(torch.load(
            f"model/subenv_{model_name}/mnist_cnn.pt"))

        losses = []
        accuracies = []
        for data_idx in num_model_4_indices:
            print(data_idx)
            td, vd = concat_data(data_idx, mode="dataset")
            tdl = DataLoader(td, batch_size=64, shuffle=True)
            vdl = DataLoader(vd, batch_size=64, shuffle=True)
            learner = Learner(tdl, vdl, lr=0.001, log_interval=100, use_cuda=use_cuda)
            learner.model = model
            loss, accuracy = learner._test()
            losses.append(loss)
            accuracies.append(accuracy)
            del learner
        writer.writerow([f"{model_name} model loss"] + losses)
        writer.writerow([f"{model_name} model accuracy"] + accuracies)
