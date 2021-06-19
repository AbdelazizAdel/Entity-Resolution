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
    
    
class SoftNLLLoss(nn.NLLLoss):
    """A soft version of negative log likelihood loss with support for label smoothing.
    Effectively equivalent to PyTorch's :class:`torch.nn.NLLLoss`, if `label_smoothing`
    set to zero. While the numerical loss values will be different compared to
    :class:`torch.nn.NLLLoss`, this loss results in the same gradients. This is because
    the implementation uses :class:`torch.nn.KLDivLoss` to support multi-class label
    smoothing.
    Args:
        label_smoothing (float):
            The smoothing parameter :math:`epsilon` for label smoothing. For details on
            label smoothing refer `this paper <https://arxiv.org/abs/1512.00567v1>`__.
        weight (:class:`torch.Tensor`):
            A 1D tensor of size equal to the number of classes. Specifies the manual
            weight rescaling applied to each class. Useful in cases when there is severe
            class imbalance in the training set.
        num_classes (int):
            The number of classes.
        size_average (bool):
            By default, the losses are averaged for each minibatch over observations **as
            well as** over dimensions. However, if ``False`` the losses are instead
            summed. This is a keyword only parameter.
    """

    def __init__(self, label_smoothing=0, weight=None, num_classes=2, **kwargs):
        super(SoftNLLLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes

        assert label_smoothing >= 0.0 and label_smoothing <= 1.0

        self.criterion = nn.KLDivLoss(**kwargs)

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.label_smoothing / (self.num_classes - 1))
        one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)

        if self.weight is not None:
            one_hot.mul_(self.weight)

        return self.criterion(input, one_hot)

