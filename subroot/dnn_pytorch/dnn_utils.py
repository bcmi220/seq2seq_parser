import numpy as np
import torch
from argparse import ArgumentParser

parser = ArgumentParser(add_help=False)
parser.add_argument(
    "--gpu", default="0",
    type=int, help="set 1 to use gpu."
)
args = parser.parse_known_args()

# set global torch tensor variables. default is using cpu
if args[0].gpu == 1:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def init_variable(shape):
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return value


def init_param(layer):
    """
    randomly initialize parameters of the given layer
    """
    for p in layer.parameters():
        p.data = torch.from_numpy(init_variable(p.size())).type(FloatTensor)

    return layer


def log_sum_exp(x, dim=None):
    """
    Sum probabilities in the log-space.
    """
    xmax, _ = x.max(dim=dim, keepdim=True)
    xmax_, _ = x.max(dim=dim)
    # return xmax_
    return xmax_ + torch.log(torch.exp(x - xmax).sum(dim=dim))


def sequence_mask(batch_len, max_len=None):
    if not max_len:
        max_len = np.max(batch_len)

    mask = np.zeros((len(batch_len), max_len))
    for i in range(len(batch_len)):
        mask[i, range(batch_len[i])] = 1

    return mask
