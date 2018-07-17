import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_temporal_pooling.modules.roi_temporal_pool import _RoITemporalPooling
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss

DEBUG = False

class _TDCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, class_agnostic):
        super(_TDCNN, self).__init__()
        #self.classes = classes
        self.class_agnostic = class_agnostic
        self.n_classes = 2 if class_agnostic else cfg.NUM_CLASSES
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_twin = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_temporal_pool = _RoITemporalPooling(cfg.POOLING_LENGTH, cfg.POOLING_HEIGHT, cfg.POOLING_WIDTH, cfg.DEDUP_TWINS)

    def forward(self, video_data, gt_twins):
        batch_size = video_data.size(0)

        gt_twins = gt_twins.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(video_data)
        # feed base feature map tp RPN to obtain rois
        # return rois, [rois_score], rpn_cls_prob, rpn_twin_pred, self.rpn_loss_cls, self.rpn_loss_twin, rpn_label
        rois, _, _, rpn_loss_cls, rpn_loss_twin, _, _ = self.RCNN_rpn(base_feat, gt_twins)
        # if it is training phrase, then use ground truth twins for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_twins)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_twin = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_temporal_pool(base_feat, rois.view(-1,3))
            if DEBUG:
                print("tdcnn.py--pooled_feat.shape1 {}".format(pooled_feat.shape))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        # compute twin offset
        twin_pred = self.RCNN_twin_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            twin_pred_view = twin_pred.view(twin_pred.size(0), int(twin_pred.size(1) / 2), 2)
            twin_pred_select = torch.gather(twin_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 2))
            twin_pred = twin_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, dim=1)

        if DEBUG:
            print("base_feat.shape {}".format(base_feat.shape))
            print("tdcnn.py--pooled_feat2.shape {}".format(pooled_feat.shape))
            #print(pooled_feat.data.cpu().numpy())
            print("tdcnn.py--cls_score.shape {}".format(cls_score.shape))
            #print(cls_score.data.cpu().numpy())

        RCNN_loss_cls = 0
        RCNN_loss_twin = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_twin = _smooth_l1_loss(twin_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        twin_pred = twin_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, twin_pred, rpn_loss_cls, rpn_loss_twin, RCNN_loss_cls, RCNN_loss_twin, rois_label

    def _init_weights(self):
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
        self.RCNN_rpn.init_weights()
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_twin_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
