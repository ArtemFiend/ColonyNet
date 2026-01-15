import torch
import torch.nn.functional as F

def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    # logits/targets: B,1,H,W
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    num = 2.0 * (probs * targets).sum(dim=1) + eps
    den = probs.sum(dim=1) + targets.sum(dim=1) + eps
    return 1.0 - (num / den).mean()

def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor):
    return F.binary_cross_entropy_with_logits(logits, targets)

def loss_total(pred: dict, y_sem, y_center, y_boundary):
    l_sem = dice_loss_with_logits(pred["sem"], y_sem) + bce_with_logits(pred["sem"], y_sem)
    l_center = bce_with_logits(pred["center"], y_center)
    l_boundary = bce_with_logits(pred["boundary"], y_boundary)
    total = l_sem + 0.5*l_center + 0.5*l_boundary
    return total, {"l_sem": l_sem.item(), "l_center": l_center.item(), "l_boundary": l_boundary.item()}
