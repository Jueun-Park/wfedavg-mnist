import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def target_generator(): return torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, 4, 2),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32, 64, 3),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 64, 3),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(3, 2),
    torch.nn.Flatten(),
    torch.nn.Linear(1024, 64),
)


def predictor_generator(): return torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, 4, 2),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32, 32, 3),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32, 32, 3),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(3, 2),
    torch.nn.Flatten(),
    torch.nn.Linear(512, 64),
)


def generator_generator(): return torch.nn.Sequential(
    torch.nn.Linear(128, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 784),
    torch.nn.Sigmoid(),
)

def discriminator_generator(): return torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 1),
    torch.nn.Sigmoid(),
)

if __name__ == "__main__":
    batch_size = 32
    target = target_generator()
    predictor = predictor_generator()
    generator = generator_generator()
    discriminator = discriminator_generator()
    x = torch.randn((batch_size, 1, 28, 28))  # mnist data

    print(target(x).shape)
    print(predictor(x).shape)

    noise = torch.randn((batch_size, 128))
    fake = generator(noise)
    print(fake.shape)
    print(fake.reshape([batch_size, 28, 28]).shape)
    print(discriminator(x).shape)
