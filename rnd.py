from numpy import array
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
from running_mean_std import RunningMeanStd
from net import target_generator, predictor_generator


class RandomNetworkDistillation:
    def __init__(self, log_interval=10, lr=1e-5, use_cuda=False):
        self.predictor = predictor_generator()
        self.target = target_generator()
        for param in self.target.parameters():
            param.requires_grad = False
        self.target.eval()

        self.log_interval = log_interval
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=lr)
        self.loss_function = torch.nn.MSELoss(reduction='mean')

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.target.to(self.device)
        self.predictor.to(self.device)

        self.running_stats = RunningMeanStd()

    def set_data(self, train_tensor, test_tensor):
        train_target_tensor = self.target(train_tensor)
        train_dataset = TensorDataset(train_tensor, train_target_tensor)
        self.train_loader = DataLoader(train_dataset)

        test_target_tensor = self.target(test_tensor)
        test_dataset = TensorDataset(test_tensor, test_target_tensor)
        self.test_loader = DataLoader(test_dataset)
        return

    def learn(self, epochs):
        for epoch in range(epochs):
            self._train(epoch)
            test_loss = self._test()
        return test_loss

    def _train(self, epoch):
        self.predictor.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.predictor(data)
            loss = self.loss_function(output, target)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx/len(self.train_loader):.0f}%)]", end="\t")
                print(f"Loss: {loss.item():.6f}")
            self.running_stats.update(arr=array([loss.item()]))
        return

    def _test(self):
        self.predictor.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.predictor(data)
                test_loss += self.loss_function(output, target).item()
        test_loss /= len(self.test_loader.dataset)
        print(f"\nTest set: Average loss: {test_loss:.4f}\n")
        return test_loss

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
        return

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
        return


if __name__ == "__main__":
    train_x = torch.randn((64, 28, 28))
    test_x = torch.randn((32, 28, 28))
    rnd = RandomNetworkDistillation(log_interval=50)
    rnd.set_data(train_x, test_x)
    rnd.learn(25)
