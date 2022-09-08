'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_ch=3, num_classes=10, stride=2):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_ch, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        a = 5 if num_ch == 3 else 4
        activations_size = 16 * a ** 2
        if stride == 1 and num_ch == 3:
            activations_size = 9216
        self.fc1   = nn.Linear(activations_size, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, stride)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, stride)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNet2(nn.Module):
    def __init__(self, num_ch=3, num_classes=10):
        super(LeNet2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(num_ch, 6, 5), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2))
        a = 5 if num_ch == 3 else 4
        self.fc1   = nn.Sequential(nn.Flatten(), nn.Linear(16 * a ** 2, 120), nn.ReLU())
        self.fc2   = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
