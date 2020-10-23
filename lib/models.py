from torch import nn


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3),  # 20 x 26 x 26
            nn.ReLU(),
            nn.Conv2d(20, 30, kernel_size=3),  # 30 x 24 x 24
            nn.ReLU(),
            nn.Conv2d(30, 20, kernel_size=3),  # 20 x 22 x 22
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 20*22*22 x 1
            nn.Linear(20*22*22, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x
