
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, normalize=None):
    self._roidb = roidb
    #self._num_classes = num_classes
    self.normalize = normalize
    # self.batch_size = batch_size

  def __getitem__(self, index):
    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index]]
    blobs = get_minibatch(minibatch_db)
    data = torch.from_numpy(blobs['data'])
    length, height, width = data.size(-3), data.size(-2), data.size(-1)
    data = data.contiguous().view(3, length, height, width)
    if cfg.TRAIN.HAS_RPN or cfg.TEST.HAS_RPN:
        gt_windows = torch.from_numpy(blobs['gt_windows'])
        #num_twin = gt_windows.size()
        #gt_windows.view(3)
        #print("data {}".format(data.shape))
        #print("gt_windows {}".format(gt_windows.shape))
        return data, gt_windows
    else: # not using RPN
        raise NotImplementedError

  def __len__(self):
    return len(self._roidb)
