from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss, mask_rpn_losses

import numpy as np
import math
import pdb
import time

DEBUG=False

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din, out_scores=False):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.out_scores = out_scores
        self.mask_upsample_rate = 1

        # define the convrelu layers processing input feature map
        self.RPN_Conv1 = nn.Conv3d(self.din, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True)
        self.RPN_Conv2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True)
        self.RPN_output_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * 2 # 2(bg/fg) * 10 (anchors)
        self.RPN_cls_score = nn.Conv3d(512, self.nc_score_out, 1, 1, 0)

        # define anchor twin offset prediction layer
        self.nc_twin_out = len(self.anchor_scales) * 2 # 2(coords) * 10 (anchors)
        self.RPN_twin_pred = nn.Conv3d(512, self.nc_twin_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.out_scores)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales)

        self.rpn_loss_cls = 0
        self.rpn_loss_twin = 0
        self.rpn_loss_mask = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3],
            input_shape[4]
        )
        return x

    def forward(self, base_feat, gt_twins):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv1(base_feat), inplace=True)
        rpn_conv2 = F.relu(self.RPN_Conv2(rpn_conv1), inplace=True)
        rpn_output_pool = self.RPN_output_pool(rpn_conv2) # (1,512,96,1,1)

        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_output_pool)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        #print("rpn_cls_score_reshape: {}".format(rpn_cls_score_reshape.shape))
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        #print("rpn_cls_prob: {}".format(rpn_cls_prob.shape))

        # get rpn offsets to the anchor twins
        rpn_twin_pred = self.RPN_twin_pred(rpn_output_pool)
        #print("rpn_twin_pred: {}".format(rpn_twin_pred.shape))

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        #rois = self.RPN_proposal((rpn_cls_prob.data, rpn_twin_pred.data, cfg_key))
        if self.out_scores:
            rois, rois_score = self.RPN_proposal((rpn_cls_prob.data, rpn_twin_pred.data, cfg_key))
        else:
            rois = self.RPN_proposal((rpn_cls_prob.data, rpn_twin_pred.data, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_twin = 0
        self.rpn_loss_mask = 0
        self.rpn_label = None

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_twins is not None
            # rpn_data = [label_targets, twin_targets, twin_inside_weights, twin_outside_weights]
            # label_targets: (batch_size, 1, A * length, height, width)
            # twin_targets: (batch_size, A*2, length, height, width), the same as twin_inside_weights and twin_outside_weights
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_twins))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, 2)
            self.rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(self.rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            self.rpn_label = torch.index_select(self.rpn_label.view(-1), 0, rpn_keep.data)
            self.rpn_label = Variable(self.rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, self.rpn_label)
            fg_cnt = torch.sum(self.rpn_label.data.ne(0))

            rpn_twin_targets, rpn_twin_inside_weights, rpn_twin_outside_weights = rpn_data[1:]

            # compute twin regression loss
            rpn_twin_inside_weights = Variable(rpn_twin_inside_weights)
            rpn_twin_outside_weights = Variable(rpn_twin_outside_weights)
            rpn_twin_targets = Variable(rpn_twin_targets)

            self.rpn_loss_twin = _smooth_l1_loss(rpn_twin_pred, rpn_twin_targets, rpn_twin_inside_weights,
                                                            rpn_twin_outside_weights, sigma=3, dim=[1,2,3,4])

        if self.out_scores:
            return rois, rois_score, rpn_cls_prob, rpn_twin_pred, self.rpn_loss_cls, self.rpn_loss_twin, self.rpn_label, self.rpn_loss_mask
        else:
            return rois, rpn_cls_prob, rpn_twin_pred, self.rpn_loss_cls, self.rpn_loss_twin, self.rpn_label, self.rpn_loss_mask

    def init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RPN_Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RPN_twin_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self.init_weights()
        
    def generate_mask_label(self, gt_twins, feat_len):
        """ 
        gt_twins will be (batch_size, n, 3), where each gt will be (x1, x2, class_id)
        # feat_len is the length of mask-task features, self.feat_stride * feat_len = video_len
        # according: self.feat_stride, and upsample_rate
        # mask will be (batch_size, feat_len), -1 -- ignore, 1 -- fg, 0 -- bg
        """
        batch_size = gt_twins.size(0)
        mask_label = torch.zeros(batch_size, feat_len).type_as(gt_twins)
        for b in range(batch_size):
           single_gt_twins = gt_twins[b]
           single_gt_twins[:, :2] = (single_gt_twins[:, :2] / self.feat_stride).int()
           twins_start = single_gt_twins[:, 0]
           _, indices = torch.sort(twins_start)
           single_gt_twins = torch.index_select(single_gt_twins, 0, indices).long().cpu().numpy()

           starts = np.minimum(np.maximum(0, single_gt_twins[:,0]), feat_len-1)
           ends = np.minimum(np.maximum(0, single_gt_twins[:,1]), feat_len)
           for x in zip(starts, ends):
              mask_label[b, x[0]:x[1]+1] = 1

        return mask_label
