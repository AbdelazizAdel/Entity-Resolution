import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, float): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction='none', weight=self.alpha)
        pt = F.softmax(input, dim=1)[range(len(target)), target]
        fl = ((1 - pt) ** self.gamma) * ce
        if(self.reduction == 'mean'):
            w = torch.tensor([self.alpha[i] for i in target])
            return fl.sum() / w.sum()
        if(self.reduction == 'sum'):
            return fl.sum()
        return fl