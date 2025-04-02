import torch
from tqdm import tqdm

def attack_success_rate(model, attack_fn, loader, eps, targeted=False, k=0, eps_step=0.0):
    model.eval()
    total = 0
    success = 0

    for images, labels in tqdm(loader, desc=f"{attack_fn.__name__}", ncols=150):
        images, labels = images.cuda(), labels.cuda()

        # 원본 예측
        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(dim=1)

        # 공격 이전 정답인 샘플만 공격 대상으로 이용
        correct_mask = preds == labels
        if correct_mask.sum() == 0:
            continue

        # Targeted인 경우 타겟 클래스 설정
        if targeted:
            target_labels = (labels + torch.randint(1, 9, labels.shape).cuda()) % 10
            x_adv = attack_fn(model, images, target_labels, k, eps, eps_step) if k > 0 else attack_fn(model, images, target_labels, eps)
        else:
            x_adv = attack_fn(model, images, labels, k, eps, eps_step) if k > 0 else attack_fn(model, images, labels, eps)

        # 공격 후 예측
        with torch.no_grad():
            adv_preds = model(x_adv).argmax(dim=1)

        total += correct_mask.sum().item()

        if targeted:
            # Targeted인 경우 Target_label로 예측되었는지 확인
            success += (adv_preds[correct_mask] == target_labels[correct_mask]).sum().item()
        else:
            # Untargeted인 경우 클래스가 변화되었는지 확인
            success += (adv_preds[correct_mask] != labels[correct_mask]).sum().item()

    rate = 100 * success / total
    kind = "Targeted" if targeted else "Untargeted"
    print(f"{kind} Attack Success Rate: {rate:.2f}% ({success}/{total})\n")
