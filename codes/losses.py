import torch.nn as nn
import torch
import torch.nn.functional as F

class WeightedSupCon(nn.Module):
    def __init__(self, temperature=0.1):
        super(WeightedSupCon, self).__init__()
        self.temperature = temperature

    def forward(self, score, ref):
        mask, weight = ref
        num_pos = mask.sum(1)
        loss = - (torch.log(
            (F.softmax(score / self.temperature, dim=1))) * mask).sum(1) / num_pos
        return (loss * weight).sum()


class SupCon(nn.Module):
    """
    Supervised Contrastive Loss
    """
    def __init__(self, temperature=0.1):
        super(SupCon, self).__init__()
        self.temperature = temperature

    def forward(self, score, mask):
        num_pos = mask.sum(1)
        loss = - (torch.log(
            (F.softmax(score / self.temperature, dim=1))) * mask).sum(1) / num_pos
        return loss.mean()


CrossEntropyLoss = torch.nn.CrossEntropyLoss
BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss
BCELoss = torch.nn.BCELoss
