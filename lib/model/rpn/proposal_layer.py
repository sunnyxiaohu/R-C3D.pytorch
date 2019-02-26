from __future__ import absolute_import
# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Shiguang Wang
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .twin_transform import twin_transform_inv, clip_twins
from model.nms.nms_wrapper import nms

import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular twins (called "anchors").
    """

    def __init__(self, feat_stride, scales, out_scores=False):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(base_size=feat_stride, scales=np.array(scales))).float()
        self._num_anchors = self._anchors.size(0)
        self._out_scores = out_scores
        # TODO: add scale_ratio for video_len ??
        # rois blob: holds R regions of interest, each is a 3-tuple
        # (n, x1, x2) specifying an video batch index n and a
        # rectangle (x1, x2)
        # top[0].reshape(1, 3)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor twins centered on cell i
        #   apply predicted twin deltas at cell i to each of the A anchors
        # clip predicted twins to video
        # remove predicted twins with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = input[0][:, self._num_anchors:, :, :, :]
        twin_deltas = input[1]
        cfg_key = input[2]
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # 1. Generate proposals from twin deltas and shifted anchors
        length, height, width = scores.shape[-3:]

        if DEBUG:
            print( 'score map size: {}'.format(scores.shape))

        batch_size = twin_deltas.size(0)

        # Enumerate all shifts
        shifts = np.arange(0, length) * self._feat_stride
        shifts = torch.from_numpy(shifts.astype(float))
        shifts = shifts.contiguous().type_as(scores)

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 2) to
        # cell K shifts (K, 1, 1) to get
        # shift anchors (K, A, 2)
        # reshape to (1, K*A, 2) shifted anchors
        # expand to (batch_size, K*A, 2)
        A = self._num_anchors
        K = shifts.shape[0]
        self._anchors = self._anchors.type_as(scores)
        anchors = self._anchors.view(1, A, 2) + shifts.view(K, 1, 1)
        anchors = anchors.view(1, K * A, 2).expand(batch_size, K * A, 2)
        # Transpose and reshape predicted twin transformations to get them
        # into the same order as the anchors:
        #
        # twin deltas will be (batch_size, 2 * A, L, H, W) format
        # transpose to (batch_size, L, H, W, 2 * A)
        # reshape to (batch_size, L * H * W * A, 2) where rows are ordered by (l, h, w, a)
        # in slowest to fastest order
        twin_deltas = twin_deltas.permute(0, 2, 3, 4, 1).contiguous()
        twin_deltas = twin_deltas.view(batch_size, -1, 2)

        # Same story for the scores:
        #
        # scores are (batch_size, A, L, H, W) format
        # transpose to (batch_size, L, H, W, A)
        # reshape to (batch_size, L * H * W * A) where rows are ordered by (l, h, w, a)
        scores = scores.permute(0, 2, 3, 4, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via twin transformations
        proposals = twin_transform_inv(anchors, twin_deltas, batch_size)

        # 2. clip predicted wins to video
        proposals = clip_twins(proposals, length * self._feat_stride, batch_size)

        # 3. remove predicted twins with either length < threshold
        # assign the score to 0 if it's non keep.
        no_keep = self._filter_twins_reverse(proposals, min_size)
        scores[no_keep] = 0
        
        scores_keep = scores
        proposals_keep = proposals
        # sorted in descending order
        _, order = torch.sort(scores_keep, 1, True)
 
        #print ("scores_keep {}".format(scores_keep.shape))
        #print ("proposals_keep {}".format(proposals_keep.shape))
        #print ("order {}".format(order.shape))

        output = scores.new(batch_size, post_nms_topN, 3).zero_()

        if self._out_scores:
            output_score = scores.new(batch_size, post_nms_topN, 2).zero_()

        for i in range(batch_size):

            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # 4. sort all (proposal, score) pairs by score from highest to lowest
            # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            #print ("num_proposal: ", num_proposal)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

            if self._out_scores:
                output_score[i, :, 0] = i
                output_score[i, :num_proposal, 1] = scores_single

        if self._out_scores:
            return output, output_score
        else:
            return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_twins_reverse(self, twins, min_size):
        """get the keep index of all twins with length smaller than min_size. 
        twins will be (batch_size, C, 2), keep will be (batch_size, C)"""
        ls = twins[:, :, 1] - twins[:, :, 0] + 1
        no_keep = (ls < min_size)
        return no_keep
