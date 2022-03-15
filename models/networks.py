import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
from torch.nn.utils import spectral_norm
from torchvision import models
import layers.functional as LF
import layers.loss as LL
import math

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, threshold=0.1, patience=3)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'exp':
        exp_decay = math.exp(-0.02)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_net(net, pretrained=True, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if not isinstance(gpu_ids, list):
        pass
    elif len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if not pretrained:
        LF.init_weights(net, init_type, init_gain=init_gain)
    return net


def define_feature_extractor(net, pretrained=True, get_early_features=None):
    """Create a generator

    Parameters:
        net (string) -- the backbone architecture used for extracting features
        pretrained (bool) -- if loading pretrained weights for the feature extractor.

    Returns a feature extraction network
    """

    # loading network from torchvision
    net_in = getattr(models, net)(pretrained=pretrained)

    if net.startswith('alexnet'):
        if get_early_features:
            features = list(net_in.children())[:-get_early_features]
        else:
            features = list(net_in.children())[:-1]
    elif net.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif net.startswith('resnet'):
        if get_early_features:
            features = list(net_in.children())[:-get_early_features]
        else:
            features = list(net_in.children())[:-2]
    elif net.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif net.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(net))

    net = nn.Sequential(*features)

    return net


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return LF.l2n(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'
