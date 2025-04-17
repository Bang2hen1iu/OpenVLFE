
import torch
import torch.nn as nn
import torch.nn.functional as F

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

class CrossEntropyLoss(nn.Module):
    """
    Standard cross entropy loss function.
    classes:int | number of output dim
    smoothing:float | smooth factor, 0 for standard one-hot embedding
    dim:int | computational dimension, default for the last dim
    """
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(CrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, category_weight=None, instance_weight=None, mean=True, prior=None):
        # prior: adapt from <Long-Tail Learning via Logit Adjustment  ICLR2021>, tau=1
        if prior is not None:
            pred= F.softmax(pred + prior, dim=self.dim)
            # pred= F.softmax(pred + torch.log(prior+1e-6), dim=self.dim)
        else:
            pred = F.softmax(pred, dim=self.dim)

        log_prob = torch.log(pred + 1e-8)
        true_dist = smooth_one_hot(target, self.cls, self.smoothing)
        if category_weight is not None:
            assert category_weight.shape[0] == pred.shape[-1]
            log_prob = category_weight * log_prob

        etp = -torch.sum(true_dist * log_prob, dim=self.dim)
        if instance_weight is not None:
            assert instance_weight.shape[0] == pred.shape[0]
            etp = instance_weight * etp

        if mean:
            return torch.mean(etp)
        else:
            return etp

def entropy(p, prob=True, mean=True, prior=None):
    if prior is not None:
        p_prior = p + prior
    else:
        p_prior = p
    if prob:
        p_prior = F.softmax(p_prior, dim=-1)
        p = F.softmax(p, dim=-1)
    en = -torch.sum(p * torch.log(p_prior+1e-8), -1)
    if mean:
        return torch.mean(en)
    else:
        return en
