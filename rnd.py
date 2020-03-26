import torch
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
from running_mean_std import RunningMeanStd


class RandomNetworkDistillation():
    def __init__(self, input_size=8, learning_late=1e-4, verbose=1, use_cuda=False, tensorboard=True):
        self.target = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 64)
        )

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 128),
            torch.nn.Linear(128, 64)
        )

        self.loss_function = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=learning_late)
        for param in self.target.parameters():
            param.requires_grad = False
        self.verbose = verbose
        self.tensorboard = tensorboard
        if self.tensorboard:
            self.summary = SummaryWriter()
        self.iteration = 0

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.target.to(self.device)
        self.predictor.to(self.device)

        self.running_stats = RunningMeanStd()

    def learn(self, x, n_steps=500):
        intrinsic_reward = self.get_intrinsic_reward(x[0])
        if self.tensorboard:
            self.summary.add_scalar(
                'intrinsic-reward', intrinsic_reward, self.iteration)
        x = np.float32(x)
        x = torch.from_numpy(x).to(self.device)
        y_train = self.target(x)
        for t in range(n_steps):
            y_pred = self.predictor(x)
            loss = self.loss_function(y_pred, y_train)
            if t % 100 == 99:
                if self.verbose > 0:
                    print("timesteps: {}, loss: {}".format(t, loss.item()))
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if self.tensorboard:
                self.summary.add_scalar(
                    'loss/loss', loss.item(), self.iteration)
            self.iteration += 1
        self.running_stats.update(arr=np.array([loss.item()]))
        if self.tensorboard:
            self.summary.add_scalar('loss/running-mean',
                                    self.running_stats.mean, self.iteration)
            self.summary.add_scalar(
                'loss/running-var', self.running_stats.var, self.iteration)

    def evaluate(self, x):
        x = np.float32(x)
        x = torch.from_numpy(x).to(self.device)
        y_test = self.target(x)
        y_pred = self.predictor(x)
        loss = self.loss_function(y_pred, y_test)
        print("evaluation loss: {}".format(loss.item()))
        return loss.item()

    def get_intrinsic_reward(self, x):
        x = np.float32(x)
        x = torch.from_numpy(x).to(self.device)
        predict = self.predictor(x)
        target = self.target(x)
        intrinsic_reward = self.loss_function(
            predict, target).data.cpu().numpy()
        intrinsic_reward = (
            intrinsic_reward - self.running_stats.mean) / np.sqrt(self.running_stats.var)
        intrinsic_reward = np.clip(intrinsic_reward, -5, 5)
        return intrinsic_reward

    def save(self, path="rnd_model/", subfix=None):
        if not os.path.isdir(path):
            os.mkdir(path)
        if subfix is not None:
            subfix = "_" + subfix
        else:
            subfix = ""
        with open("{}/running_stat.pkl".format(path), 'wb') as f:
            pickle.dump(self.running_stats, f)
        torch.save(self.target.state_dict(),
                   "{}/target{}.pt".format(path, subfix))
        torch.save(self.predictor.state_dict(),
                   "{}/predictor{}.pt".format(path, subfix))

    def load(self, path="rnd_model/", subfix=None):
        if subfix is not None:
            subfix = "_" + subfix
        else:
            subfix = ""
        with open("{}/running_stat.pkl".format(path), 'rb') as f:
            self.running_stats = pickle.load(f)
        self.target.load_state_dict(torch.load(
            "{}/target{}.pt".format(path, subfix), map_location=torch.device(self.device)))
        self.predictor.load_state_dict(torch.load(
            "{}/predictor{}.pt".format(path, subfix), map_location=torch.device(self.device)))

    def set_to_inference(self):
        self.target.eval()
        self.predictor.eval()
