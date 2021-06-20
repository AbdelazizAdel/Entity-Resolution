import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

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

    def __init__(self, label_smoothing=0, weight=None, num_classes=2, **kwargs):
        super(SoftNLLLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes

        assert label_smoothing >= 0.0 and label_smoothing <= 1.0

        self.criterion = nn.KLDivLoss(**kwargs, reduction='batchmean')

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.label_smoothing / (self.num_classes - 1))
        one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)

        if self.weight is not None:
            one_hot.mul_(self.weight)

        return self.criterion(input, one_hot)
    

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):          
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def plot_stats(train_stats, val_stats):
    fig, (ax1, ax2, ax3, ax4) = plt.subplot(2,2)
    plot_graph(ax1, train_stats[0], val_stats[0], 'accuracy')
    plot_graph(ax2, train_stats[1], val_stats[1], 'recall')
    plot_graph(ax3, train_stats[2], val_stats[2], 'precision')
    plot_graph(ax4, train_stats[3], val_stats[3], 'f1')
    
def plot_graph(ax, train_data, val_data, label):
    x_axis = np.arange(len(train_data))
    ax.plot(x_axis, train_data, label='train')
    ax.plot(x_axis, val_data, label='validation')
    ax.set_xlabel('epoch')
    ax.set_ylabel(label)