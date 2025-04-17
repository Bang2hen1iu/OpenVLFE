import math

def inv_lr_scheduler(epoch_num, gamma=10, power=0.75, max_epoch=10000):
    factor = (1 + gamma * min(1.0, epoch_num / max_epoch)) ** (-power)
    return factor

def alpha_scheduler(epoch, nepochs):
    p = float(epoch) / nepochs
    o = -10
    alpha = 2. / (1. + math.exp(o * p)) - 1.0
    return alpha