import torch
import  torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inputLayer = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.hiddenLayer = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.outputLayer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.relu(self.inputLayer(input))
        for _ in range(19):
            out = self.relu(self.hiddenLayer(out))

        out = self.relu(self.outputLayer(out))

        return torch.add(out, input)
