import torch
import torch.nn as nn
from torchsummary import summary


class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):

    def __init__(self, features=[1, 16, 32, 64, 128]):
        super().__init__()
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature)
            )
            in_channels = feature
        self.model = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.softmax(x)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = CNN().to(device)
    summary(cnn, (1, 64, 44))
