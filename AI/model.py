import torch.nn as nn
import copy

class DDQN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
    
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
           p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)