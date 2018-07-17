# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Shiguang Wang
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, video_list_to_blob
from model.utils.transforms import GroupMultiScaleCrop
import pdb

DEBUG = False

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_videos = len(roidb)
    # Sample random scales to use for each video in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.LENGTH),
                                    size=num_videos)

    # Get the input video blob, formatted for caffe
    video_blob = _get_video_blob(roidb, random_scale_inds)
    blobs = {'data': video_blob}
    # TODO: match video_blob and gt_windows, fix the bug when training and test procedure mismatch
    if cfg.TRAIN.HAS_RPN or cfg.TEST.HAS_RPN:
        assert len(roidb) == 1, "Single batch only"
        # gt windows: (x1, x2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_windows = np.empty((len(gt_inds), 3), dtype=np.float32)
        gt_windows[:, 0:2] = roidb[0]['wins'][gt_inds, :]
        gt_windows[:, -1] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_windows'] = gt_windows
    else: # not using RPN
        raise NotImplementedError

    if cfg.AGNOSTIC:
        blobs['gt_windows'][:, -1] = 1

    return blobs

def _get_video_blob(roidb, scale_inds):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """
    processed_videos = []
    video_scales = []
    for i,item in enumerate(roidb):
      # just one scale implementated
      video_length = cfg.TRAIN.LENGTH[scale_inds[0]]  
      video = np.zeros((video_length, cfg.TRAIN.CROP_SIZE,
                        cfg.TRAIN.CROP_SIZE, 3))
      #if cfg.INPUT == 'video':
      j = 0
      #random_idx = [np.random.randint(cfg.TRAIN.FRAME_SIZE[1]-cfg.TRAIN.CROP_SIZE),
      #                np.random.randint(cfg.TRAIN.FRAME_SIZE[0]-cfg.TRAIN.CROP_SIZE)]
      image_w, image_h, crop_w, crop_h = cfg.TRAIN.FRAME_SIZE[1], cfg.TRAIN.FRAME_SIZE[0], cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE
      offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h) 
      random_idx = offsets[ npr.choice(len(offsets)) ]
      if DEBUG:
        print ("offsets: {}, random_idx: {}".format(offsets, random_idx))
      for video_info in item['frames']:
        prefix = item['fg_name'] if video_info[0] else item['bg_name']
        step = video_info[3] if cfg.INPUT == 'video' else 1
        for idx in range(video_info[1], video_info[2], video_info[3]):
          frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))
          frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]), cfg.TRAIN.CROP_SIZE, random_idx)
          if item['flipped']:
            frame = frame[:, ::-1, :]

          if DEBUG:
            cv2.imshow('frame', frame/255.0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

          video[j] = frame
          j = j + 1
      # padding for the same length
      while ( j < video_length):
        video[j] = frame
        j = j + 1

      processed_videos.append(video)

    # Create a blob to hold the input images
    blob = video_list_to_blob(processed_videos)

    return blob
