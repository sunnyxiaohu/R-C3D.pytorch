#coding=utf-8
# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
import copy
import json
import cPickle
import numpy as np
import cv2
from util import *

FPS = 25
LENGTH = 768
STEP = LENGTH / 4
WINS = [LENGTH * 3] #<=> FPS = 5, WINS = [LENGTH * 1]
DATA_DIR = '/media/agwang/ShiGuangB/Datasets/Charades'
VIDEO_DIR = '/media/agwang/新加卷/DataSets/Charades'
FRAME_DIR = '/media/agwang/03c94b1e-c46c-4c7b-8d3f-47e316fdee74/home/ksnzh/Videos/action-datasets/Charades'
META_DIR = os.path.join(VIDEO_DIR, 'Charades_meta')

USE_FLIPPED = False

def generate_roi(video, start, end, stride, split):
  tmp = {}
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
  tmp['bg_name'] = os.path.join(FRAME_DIR, split, video)
  tmp['fg_name'] = os.path.join(FRAME_DIR, split, video)
  if not os.path.isfile(os.path.join(FRAME_DIR, split, video, 'image_' + str(end-1).zfill(5) + '.jpg')):
    print os.path.join(FRAME_DIR, split, video, 'image_' + str(end-1).zfill(5) + '.jpg')
    raise
  return tmp

def generate_roidb(split):
  VIDEO_PATH = os.path.join(FRAME_DIR, split)
  video_list = os.listdir(VIDEO_PATH)
  roidb = []
  for i,vid in enumerate(video_list):
    #print i
    length = len(os.listdir(os.path.join(VIDEO_PATH, vid)))

    for win in WINS:
      stride = win / LENGTH
      step = stride * STEP
      # Forward Direction
      for start in xrange(0, max(1, length - win + 1), step):
        end = min(start + win, length)
        assert end <= length

        # Add data
        tmp = generate_roi(vid, start, end, stride, split)
        roidb.append(tmp)

        if USE_FLIPPED:
          flipped_tmp = copy.deepcopy(tmp)
          flipped_tmp['flipped'] = True
          roidb.append(flipped_tmp)

      # Backward Direction
      # for end in xrange(length, win, - step):
      for end in xrange(length, win-1, - step):
        start = end - win
        assert start >= 0

        # Add data
        tmp = generate_roi(vid, start, end, stride, split)
        roidb.append(tmp)

        if USE_FLIPPED:
          flipped_tmp = copy.deepcopy(tmp)
          flipped_tmp['flipped'] = True
          roidb.append(flipped_tmp)

  return roidb
      
val_roidb = generate_roidb('test')
print len(val_roidb)
  
print "Save dictionary"
cPickle.dump(val_roidb, open('val_data_8fps.pkl','w'), cPickle.HIGHEST_PROTOCOL)
