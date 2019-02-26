from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Shiguang Wang
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from ..utils.config import cfg
from .twin_transform import twins_overlaps_batch, twin_transform_batch
import pdb

DEBUG = False

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.TWIN_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_MEANS)
        self.TWIN_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.TWIN_NORMALIZE_STDS)
        self.TWIN_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.TWIN_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_twins):
        # GT twins (batch_size, N, 3), each row of gt twin contains (x1, x2, label)
        # all_rois (batch_size, K, 3), each row of all_rois contains (video_idx, x1, x2)
        self.TWIN_NORMALIZE_MEANS = self.TWIN_NORMALIZE_MEANS.type_as(gt_twins)
        self.TWIN_NORMALIZE_STDS = self.TWIN_NORMALIZE_STDS.type_as(gt_twins)
        self.TWIN_INSIDE_WEIGHTS = self.TWIN_INSIDE_WEIGHTS.type_as(gt_twins)

        gt_twins_append = gt_twins.new(gt_twins.size()).zero_()
        gt_twins_append[:,:,1:3] = gt_twins[:,:,:2]
        
        # Include ground-truth twins in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_twins_append], 1)
        # print("gt_twins: ", gt_twins.size(), "all_rois: ", all_rois.size())
        
        num_videos = 1
        rois_per_video = int(cfg.TRAIN.BATCH_SIZE / num_videos)
        fg_rois_per_video = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_video))
        fg_rois_per_video = 1 if fg_rois_per_video == 0 else fg_rois_per_video

        labels, rois, twin_targets, twin_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_twins, fg_rois_per_video,
            rois_per_video, self._num_classes)

        twin_outside_weights = (twin_inside_weights > 0).float()

        return rois, labels, twin_targets, twin_inside_weights, twin_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_twin_regression_labels_pytorch(self, twin_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (twin_target_data) are stored in a
        compact form b x N x (tx, tl)

        This function expands those targets into the 2-of-2*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            twin_target (ndarray): b x N x 2K blob of regression targets
            twin_inside_weights (ndarray): b x N x 2K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_video = labels_batch.size(1)
        clss = labels_batch
        twin_targets = twin_target_data.new(batch_size, rois_per_video, 2).zero_()
        twin_inside_weights = twin_target_data.new(twin_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                twin_targets[b, ind, :] = twin_target_data[b, ind, :]
                twin_inside_weights[b, ind, :] = self.TWIN_INSIDE_WEIGHTS

        return twin_targets, twin_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an video."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 2
        assert gt_rois.size(2) == 2

        batch_size = ex_rois.size(0)
        rois_per_video = ex_rois.size(1)

        targets = twin_transform_batch(ex_rois, gt_rois)

        if cfg.TRAIN.TWIN_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.TWIN_NORMALIZE_MEANS.expand_as(targets))
                        / self.TWIN_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _sample_rois_pytorch(self, all_rois, gt_twins, fg_rois_per_video, rois_per_video, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # assume all_rois(batch_size, N, 3) and gt_wins(batch_size, K, 3), respectively, overlaps will be (batch_size, N, K)
        overlaps = twins_overlaps_batch(all_rois, gt_twins)
        # find max_overlaps for each dt: (batch_size, N)
        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_twins_per_video = overlaps.size(2)

        offset = torch.arange(0, batch_size)*gt_twins.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment
        labels = gt_twins[:,:,2].contiguous().view(-1)[offset.view(-1)].view(batch_size, -1)

        labels_batch = labels.new(batch_size, rois_per_video).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_video, 3).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_video, 3).zero_()
        # Guard against the case when an video has fewer than max_fg_rois_per_video
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            if DEBUG:
                print ("fg_num_rois: {}, bg_num_rois: {}".format(fg_num_rois, bg_num_rois))

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_video = min(fg_rois_per_video, fg_num_rois)
                
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault. 
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_twins).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_video]]

                # sampling bg
                bg_rois_per_this_video = rois_per_video - fg_rois_per_this_video

                # Seems torch.rand has a bug, it will generate very large number and make an error. 
                # We use numpy rand instead. 
                #rand_num = (torch.rand(bg_rois_per_this_video) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_video) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_twins).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_video) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_video) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_twins).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_video = rois_per_video
                bg_rois_per_this_video = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_video) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_video) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_twins).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_video = rois_per_video
                fg_rois_per_this_video = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
                
            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_video < rois_per_video:
                labels_batch[i][fg_rois_per_this_video:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            gt_rois_batch[i] = gt_twins[i][gt_assignment[i][keep_inds]]

        twin_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:3], gt_rois_batch[:,:,:2])

        twin_targets, twin_inside_weights = \
                self._get_twin_regression_labels_pytorch(twin_target_data, labels_batch, num_classes)

        return labels_batch, rois_batch, twin_targets, twin_inside_weights
