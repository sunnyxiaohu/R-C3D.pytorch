
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
    def __init__(self, roidb, normalize=None, phase='train'):
        self._roidb = roidb
        self.max_num_box = cfg.MAX_NUM_GT_TWINS
        self.normalize = normalize
        self.phase = phase

    def __getitem__(self, index):
        # get the anchor index for current sample index
        item = self._roidb[index]
        blobs = get_minibatch([item], self.phase)
        data = torch.from_numpy(blobs['data'])
        length, height, width = data.shape[-3:]
        data = data.contiguous().view(3, length, height, width)

        gt_windows = torch.from_numpy(blobs['gt_windows'])
        gt_windows_padding = gt_windows.new(self.max_num_box, gt_windows.size(1)).zero_()
        num_gt = min(gt_windows.size(0), self.max_num_box)
        gt_windows_padding[:num_gt, :] = gt_windows[:num_gt]
        
        if self.phase == 'test':
            video_info = ''
            for key, value in item.items():
                video_info = video_info + " {}: {}\n".format(key, value)
            # drop the last "\n"
            video_info = video_info[:-1]
            return data, gt_windows_padding, num_gt, video_info
        else:
            return data, gt_windows_padding, num_gt     

    def __len__(self):
        return len(self._roidb)
