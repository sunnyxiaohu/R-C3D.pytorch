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
import subprocess
import numpy as np
import cv2
from util import *

FPS = 25
LENGTH = 768
WINS = [LENGTH * 8]
#LENGTH = 192
#WINS = [LENGTH * 32]
min_length = 3
overlap_thresh = 0.7
STEP = LENGTH / 4
META_FILE = './activity_net.v1-3.min.json'
data = json.load(open(META_FILE))
FRAME_DIR = '/media/F/ActivityNet/frames_'+str(FPS)

print ('Generate Classes')
classes = generate_classes(data)

print ('Generate Training Segments')
train_segment = generate_segment('training', data, classes, FRAME_DIR)


def generate_roi(rois, video, start, end, stride, split):
    tmp = {}
    tmp['wins'] = ( rois[:,:2] - start ) / stride
    tmp['durations'] = tmp['wins'][:,1] - tmp['wins'][:,0]
    tmp['gt_classes'] = rois[:,2]
    tmp['max_classes'] = rois[:,2]
    tmp['max_overlaps'] = np.ones(len(rois))
    tmp['flipped'] = False
    tmp['frames'] = np.array([[0, start, end, stride]])
    tmp['bg_name'] = os.path.join(FRAME_DIR, split, video)
    tmp['fg_name'] = os.path.join(FRAME_DIR, split, video)
    if not os.path.isfile(os.path.join(FRAME_DIR, split, video, 'image_' + str(end-1).zfill(5) + '.jpg')):
        print (os.path.join(FRAME_DIR, split, video, 'image_' + str(end-1).zfill(5) + '.jpg'))
        raise
    return tmp

def generate_roidb(split, segment):
    VIDEO_PATH = os.path.join(FRAME_DIR, split)
    video_list = set(os.listdir(VIDEO_PATH))
    duration = []
    roidb = []
    for vid in segment:
        if vid in video_list:
            length = len(os.listdir(os.path.join(VIDEO_PATH, vid)))
            db = np.array(segment[vid])
            if len(db) == 0:
                continue
            db[:,:2] = db[:,:2] * FPS

            for win in WINS:
                stride = int(win / LENGTH)
                step = int(stride * STEP)
                
                # Forward Direction
                for start in range(0, max(1, length - win + 1), step):
                    end = min(start + win, length)
                    assert end <= length
                    # No overlap between gt and dt
                    rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] <= start))]

                    # Remove duration less than min_length
                    if len(rois) > 0:
                        duration = rois[:,1] - rois[:,0]
                        rois = rois[duration >= min_length]

                    # Remove overlap(for gt) less than overlap_thresh
                    if len(rois) > 0:
                        time_in_wins = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0
                        overlap = time_in_wins / (rois[:,1] - rois[:,0])
                        assert min(overlap) >= 0
                        assert max(overlap) <= 1
                        rois = rois[overlap >= overlap_thresh]

                    # Append data
                    if len(rois) > 0:
                        rois[:,0] = np.maximum(start, rois[:,0])
                        rois[:,1] = np.minimum(end, rois[:,1])
                        tmp = generate_roi(rois, vid, start, end, stride, split)
                        roidb.append(tmp)
                        if USE_FLIPPED:
                               flipped_tmp = copy.deepcopy(tmp)
                               flipped_tmp['flipped'] = True
                               roidb.append(flipped_tmp)

                # Backward Direction
                for end in range(length, win-1, - step):
                    start = end - win
                    assert start >= 0
                    rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] <= start))]

                    # Remove duration less than min_length
                    if len(rois) > 0:
                        duration = rois[:,1] - rois[:,0]
                        rois = rois[duration > min_length]

                    # Remove overlap less than overlap_thresh
                    if len(rois) > 0:
                        time_in_wins = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0
                        overlap = time_in_wins / (rois[:,1] - rois[:,0])
                        assert min(overlap) >= 0
                        assert max(overlap) <= 1
                        rois = rois[overlap > overlap_thresh]

                    # Append data
                    if len(rois) > 0:
                        rois[:,0] = np.maximum(start, rois[:,0])
                        rois[:,1] = np.minimum(end, rois[:,1])
                        tmp = generate_roi(rois, vid, start, end, stride, split)
                        roidb.append(tmp)
                    if USE_FLIPPED:
                           flipped_tmp = copy.deepcopy(tmp)
                           flipped_tmp['flipped'] = True
                           roidb.append(flipped_tmp)

    return roidb


USE_FLIPPED = True      
train_roidb = generate_roidb('training', train_segment)
print (len(train_roidb))
print ("Save dictionary")
pickle.dump(train_roidb, open('train_data_{}fps_flipped.pkl'.format(FPS),'wb'), pickle.HIGHEST_PROTOCOL)
