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
import pickle
import numpy as np
import cv2
from util import *

FPS = 25
LENGTH = 768
WINS = [LENGTH * 8]
#LENGTH = 192
#WINS = [LENGTH * 32]
STEP = LENGTH / 4
FRAME_DIR = '/media/F/ActivityNet/frames_'+str(FPS)

USE_FLIPPED = False

def generate_roi(video, start, end, stride, split):
    tmp = {}
    tmp['flipped'] = False
    tmp['frames'] = np.array([[0, start, end, stride]])
    tmp['bg_name'] = os.path.join(FRAME_DIR, split, video)
    tmp['fg_name'] = os.path.join(FRAME_DIR, split, video)
    if not os.path.isfile(os.path.join(tmp['bg_name'], 'image_' + str(end-1).zfill(5) + '.jpg')):
        print (os.path.join(tmp['bg_name'], 'image_' + str(end-1).zfill(5) + '.jpg'))
        raise
    return tmp

def generate_roidb(split):
    VIDEO_PATH = os.path.join(FRAME_DIR, split)
    video_list = os.listdir(VIDEO_PATH)
    roidb = []
    for i,vid in enumerate(video_list):
        #print i
        length = len(os.listdir(os.path.join(VIDEO_PATH, vid)))
        if (length==0):
            continue
        for win in WINS:
            stride = int(win / LENGTH)
            step = int(stride * STEP)
            # Forward Direction
            for start in range(0, max(1, length - win + 1), step):
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
            for end in range(length, win-1, - step):
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
      
val_roidb = generate_roidb('validation')
print (len(val_roidb))
  
print ("Save dictionary")
pickle.dump(val_roidb, open('val_data_{}fps.pkl'.format(FPS),'wb'), pickle.HIGHEST_PROTOCOL)
