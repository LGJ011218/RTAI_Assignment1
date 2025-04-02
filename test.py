import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from models import MNISTModel, get_pretrained_cifar10_model
from attack_methods import fgsm_targeted, fgsm_untargeted, pgd_targeted, pgd_untargeted
from attack_test import attack_success_rate

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_list = ['MNIST', 'CIFAR-10']

# MNIST와 CIFAR-10에 대해 순서대로 시행
for model_name in model_list:
    if model_name == 'MNIST':
        # MNIST 데이터셋 준비
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # 모델 초기화
        model = MNISTModel().cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 학습 진행
        print("Train MNIST Model")
        model.train()
        for epoch in tqdm(range(20), ncols=150):
            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("Training done")

    elif model_name == 'CIFAR-10':
        # CIFAR-10 데이터셋 준비
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        # 모델 로딩
        model = get_pretrained_cifar10_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 학습 진행
        print("Train CIFAR-10 Model")
        model.train()
        for epoch in tqdm(range(20), ncols=150):
            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("Training done")

    # 공격 진행 및 공격 성공률 측정
    model.eval()

    print("==========================================")
    print(f'[{model_name} Dataset]')

    # FGSM Untargeted
    attack_success_rate(model, fgsm_untargeted, test_loader, eps=0.3, targeted=False)

    # FGSM Targeted
    attack_success_rate(model, fgsm_targeted, test_loader, eps=0.3, targeted=True)

    # PGD Untargeted
    attack_success_rate(model, pgd_untargeted, test_loader, eps=0.3, eps_step=0.03, k=10, targeted=False)

    # PGD Targeted
    attack_success_rate(model, pgd_targeted, test_loader, eps=0.3, eps_step=0.03, k=10, targeted=True)