from datetime import datetime
import os
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from os import path
import sys
sys.path.append(path.abspath(path.dirname(__file__)))
from pytorchtools import EarlyStopping
from net import Net


class Learner:
    def __init__(self, train_loader, test_loader, lr=1.0, use_cuda=False, log_interval=10, tensorboard=False, tensorboard_comment=""):
        self.model = Net()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        gamma = 0.7
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)
        self.log_interval = log_interval
        if tensorboard:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs_learner', current_time+tensorboard_comment)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        # self.save(f"model/subenv_{tensorboard_comment}")
        # self.early_stopping = EarlyStopping(save_dir=f"model/subenv_{tensorboard_comment}")
        self.n_iter = 0

    def learn(self, epochs):
        for epoch in range(epochs):
            self._train(epoch)
            self._test()
            self.scheduler.step()

    def _train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.n_iter += 1
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
                if self.writer is not None:
                    self.writer.add_scalar("Loss/train", loss.item(), self.n_iter)
                    self.writer.add_scalar("Accuracy/train", 100. * batch_idx / len(self.train_loader), self.n_iter)

    def _test(self):
        self.model.eval()
        self.model.to(self.device)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        if self.writer is not None:
            self.writer.add_scalar("Loss/test", test_loss, self.n_iter)
            self.writer.add_scalar("Accuracy/test", 100. * correct / len(self.test_loader.dataset), self.n_iter)
        # self.early_stopping(test_loss, self.model)
        # if self.early_stopping.early_stop:
        #     print(">> save early stop checkpoint")
        return test_loss

    def save(self, dir_name="./model"):
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        file_path = f"{dir_name}/mnist_cnn.pt"
        torch.save(self.model.state_dict(), file_path)

    def load(self):
        pass


if __name__ == "__main__":
    from torchvision import datasets, transforms
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
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True, **kwargs)

    learner = Learner(train_loader, test_loader)
    learner.learn(2)
    learner.save()
