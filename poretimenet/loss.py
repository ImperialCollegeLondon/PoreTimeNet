import torch
from torch import nn
from torch.nn import functional as F


class SparseSegmentationLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=-1, dice_weight=1.0, ce_weight=1.0,
                 label_smoothing=0.05, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        weights = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index,
                                      label_smoothing=label_smoothing)

    def forward(self, logits, target):
        if logits.ndim != 6 or target.shape != (logits.shape[0], *logits.shape[2:]):
            raise ValueError('Expected logits (B,C,T,D,H,W) and targets (B,T,D,H,W)')
        if logits.shape[1] != self.num_classes:
            raise ValueError('Output class count does not match the loss')
        valid = target != self.ignore_index
        if not valid.any():
            raise ValueError('The batch has no supervised voxels')
        if ((target[valid] < 0) | (target[valid] >= self.num_classes)).any():
            raise ValueError('Target class indices are outside the configured class range')
        total_dice = logits.new_zeros((), dtype=torch.float32)
        total_ce = logits.new_zeros((), dtype=torch.float32)
        slots = 0
        for time in range(logits.shape[2]):
            active = valid[:, time].flatten(1).any(dim=1)
            count = int(active.sum())
            if not count:
                continue
            prediction = logits[active, :, time].float()
            truth = target[active, time].long()
            mask = truth != self.ignore_index
            safe = truth.masked_fill(~mask, 0)
            probabilities = prediction.softmax(dim=1) * mask.unsqueeze(1)
            one_hot = F.one_hot(safe, self.num_classes).permute(0, 4, 1, 2, 3).to(probabilities.dtype)
            one_hot = one_hot * mask.unsqueeze(1)
            axes = (0, 2, 3, 4)
            intersection = (probabilities * one_hot).sum(dim=axes)
            denominator = probabilities.sum(dim=axes) + one_hot.sum(dim=axes)
            dice = 1 - ((2 * intersection + 1e-5) / (denominator + 1e-5)).mean()
            total_dice = total_dice + dice * count
            total_ce = total_ce + self.ce(prediction, truth) * count
            slots += count
        dice, ce = total_dice / slots, total_ce / slots
        return self.dice_weight * dice + self.ce_weight * ce, {'dice_loss': dice.detach(), 'ce_loss': ce.detach()}
