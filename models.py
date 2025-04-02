import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# MNIST 모델
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (N, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CIFAR-10 모델
# Pytorch의 torchvision 라이브러리에서 제공하는 ImageNet에 대해 사전학습된 ResNet-18 모델 사용
def get_pretrained_cifar10_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # CIFAR-10용으로 구조 수정
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 클래스

    return model