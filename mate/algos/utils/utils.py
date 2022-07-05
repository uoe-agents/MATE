import collections
import math

import gym
import torch
from torch._C import Value
from torch.autograd import Variable
import numpy as np


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def split_batch(splits, device, dim=-1):
    def thunk(batch):
        return torch.split(batch.to(device), splits, dim=dim)
    return thunk


def split_dims(gym_spaces):
    """
    Extract dimensions for splitting
    :param gym_spaces (List[gym.spaces]): gym spaces
    :return (List[int]): splitting dims
    """
    is_box = lambda x: isinstance(x, gym.spaces.Box)
    all_box = all([is_box(space) for space in gym_spaces])
    none_box = all([not is_box(space) for space in gym_spaces])
    assert all_box or none_box

    if none_box:
        return [gym.spaces.flatdim(space) for space in gym_spaces]
    else:
        # all box --> have shape property
        return [space.shape[0] for space in gym_spaces]


def concat_shapes(shapes):
    """
    Concatenate shape of multiple shapes
    :param shapes (List[Tuple[int]]): list of shapes
    :return: concatenated shape
    """
    # all need to have same length and same shape aside from first entry
    assert len(shapes) >= 1
    shape = shapes[0]
    if not all([len(s) == len(shape) for s in shapes[1:]]):
        raise ValueError("All shapes for concatenation need to have same dimensionality.")
    if not all([s[1:] == shape[1:] for s in shapes[1:]]):
        raise ValueError("All shapes for concatenation need to have values in all dims but 0.")
    cat_dim = sum([s[0] for s in shapes])
    return tuple([cat_dim] + list(shape[1:]))


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
