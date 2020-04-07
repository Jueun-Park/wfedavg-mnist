import torch
from torch.utils.data import DataLoader
from pathlib import Path

from os import path
import sys
sys.path.append(path.abspath(path.dirname(__file__)))
from net import generator_generator, discriminator_generator


class GenerativeAdversarialNetwork:
    def __init__(self, use_cuda=False, save_path="./model/gan"):
        self.generator = generator_generator()
        self.discriminator = discriminator_generator()

        self.criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.save_path = save_path
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def set_data(self, train_tensor, test_tensor, batch_size=64):
        self.train_loader = DataLoader(
            train_tensor, shuffle=True, batch_size=batch_size)
        self.test_loader = DataLoader(
            test_tensor, shuffle=True, batch_size=batch_size)

    def learn(self, epochs):
        for epoch in range(epochs):
            self._train(epoch)
            self._test()

    def _train(self, epoch):
        self.generator.train()
        self.discriminator.train()
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)

            z = torch.randn((data.shape[0], 128))
            target_valid = torch.Tensor(data.size(0), 1).fill_(1.0)
            target_fake = torch.Tensor(data.size(0), 1).fill_(0.0)

            # train Dis
            self.d_optimizer.zero_grad()
            real_loss = self.criterion(self.discriminator(data), target_valid)
            fake_loss = self.criterion(
                self.discriminator(self.generator(z)), target_fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            self.d_optimizer.step()

            # train Gen
            self.g_optimizer.zero_grad()
            g_loss = self.criterion(self.discriminator(
                self.generator(z)), target_valid)
            g_loss.backward()
            self.g_optimizer.step()

            print(
                f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx/len(self.train_loader):.0f}%)]", end="\t")
            print(f"[D loss: {d_loss:.6f}] [G loss: {g_loss:.6f}]")

    def _test(self):
        self.generator.eval()
        self.discriminator.eval()
        test_loss = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                z = torch.randn((data.shape[0], 128))
                target_valid = torch.Tensor(data.size(0), 1).fill_(1.0)
                target_fake = torch.Tensor(data.size(0), 1).fill_(0.0)

                self.d_optimizer.zero_grad()
                real_loss = self.criterion(
                    self.discriminator(data), target_valid)
                fake_loss = self.criterion(
                    self.discriminator(self.generator(z)), target_fake)
                d_loss = (real_loss + fake_loss) / 2

                self.g_optimizer.zero_grad()
                g_loss = self.criterion(self.discriminator(
                    self.generator(z)), target_valid)

                print(f"Test set: [D loss: {d_loss}] [G loss: {g_loss}]")

    def save(self):
        torch.save(self.generator.state_dict(), f"{self.save_path}/generator.pt")
        torch.save(self.discriminator.state_dict(), f"{self.save_path}/discriminator.pt")

    def load(self):
        self.generator.load_state_dict(torch.load(f"{self.save_path}/generator.pt", map_location=torch.device(self.device)))
        self.discriminator.load_state_dict(torch.load(f"{self.save_path}/discriminator.pt", map_location=torch.device(self.device)))

    def get_discriminator_output(self, data):
        return torch.mean(self.discriminator(data))


if __name__ == "__main__":
    train_x = torch.randn((512, 1, 28, 28))
    test_x = torch.randn((64, 1, 28, 28))
    gan = GenerativeAdversarialNetwork()
    gan.set_data(train_x, test_x)
    gan.learn(2)
