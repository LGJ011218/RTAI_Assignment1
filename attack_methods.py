import torch
import torch.nn.functional as F

# FGSM Attack Methods
def fgsm_targeted(model, x, target, eps):
    # 입력 이미지 복사 및 gradient 계산을 위한 설정
    x_adv = x.clone().detach().requires_grad_(True)
    # 모델 예측 및 타겟 클래스에 대한 loss 계산
    outputs = model(x_adv)
    loss = F.cross_entropy(outputs, target)
    # 이전 gradient 초기화 및 역전파
    model.zero_grad()
    loss.backward()
    # loss를 줄이는 방향으로 perturbation 추가
    x_adv = x_adv - eps * x_adv.grad.sign()
    # 픽셀값 클램핑
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def fgsm_untargeted(model, x, label, eps):
    # 입력 이미지 복사 및 gradient 계산을 위한 설정
    x_adv = x.clone().detach().requires_grad_(True)
    # 정답 레이블 기준으로 loss 계산
    outputs = model(x_adv)
    loss = F.cross_entropy(outputs, label)
    # 이전 gradient 초기화 및 역전파
    model.zero_grad()
    loss.backward()
    # loss를 최대화하는 방향으로 perturbation 추가
    x_adv = x_adv + eps * x_adv.grad.sign()
    # 픽셀값 클램핑
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

# PGD Attack Methods
def pgd_targeted(model, x, target, k, eps, eps_step):
    # 원본 이미지 저장
    x_orig = x.clone().detach()
    # 입력 이미지 복사 및 gradient 계산을 위한 설정
    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(k):
        # 현재 예측 결과로부터 loss 계산
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, target)
        # 이전 gradient 초기화 및 역전파
        model.zero_grad()
        loss.backward()
        # loss를 줄이는 방향으로 perturbation 추가
        x_adv = x_adv - eps_step * x_adv.grad.sign()
        # perturbation의 범위 내로 제한
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        # 픽셀값 클램핑 후 다시 gradient 계산을 위해 설정
        x_adv = torch.clamp(x_adv, 0, 1).detach().requires_grad_(True)

    return x_adv

def pgd_untargeted(model, x, label, k, eps, eps_step):
    # 원본 이미지 저장
    x_orig = x.clone().detach()
    # 입력 이미지 복사 및 gradient 계산을 위한 설정
    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(k):
        # 현재 예측 결과로부터 loss 계산
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, label)
        # 이전 gradient 초기화 및 역전파
        model.zero_grad()
        loss.backward()
        # loss를 최대화하는 방향으로 perturbation 추가
        x_adv = x_adv + eps_step * x_adv.grad.sign()
        # perturbation의 범위 내로 제한
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        # 픽셀값 클램핑 후 다시 gradient 계산을 위해 설정
        x_adv = torch.clamp(x_adv, 0, 1).detach().requires_grad_(True)

    return x_adv