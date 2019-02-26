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
import os
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, video_list_to_blob
from model.utils.transforms import GroupMultiScaleCrop
import pdb
from multiprocessing import Pool, cpu_count
import threading

DEBUG = False

def get_minibatch(roidb, phase='train'):
    """Given a roidb, construct a minibatch sampled from it."""
    num_videos = len(roidb)
    assert num_videos == 1, "Single batch only"
    # Sample random scales to use for each video in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.LENGTH),
                                    size=num_videos)

    # Get the input video blob, formatted for caffe
    video_blob = _get_video_blob(roidb, random_scale_inds, phase=phase)
    blobs = {'data': video_blob}
    
    if phase != 'train':
        blobs['gt_windows'] = np.zeros((1, 3), dtype=np.float32)
        return blobs
        
    # gt windows: (x1, x2, cls)
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    gt_windows = np.empty((len(gt_inds), 3), dtype=np.float32)
    gt_windows[:, 0:2] = roidb[0]['wins'][gt_inds, :]
    gt_windows[:, -1] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_windows'] = gt_windows

    return blobs

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)
        
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
            
def prepare_im_func(prefix, random_idx, frame_idx, flipped):        
    frame_path = os.path.join(prefix, 'image_'+str(frame_idx).zfill(5)+'.jpg')
    frame = cv2.imread(frame_path)
    # process the boundary frame
    if frame is None:          
        frames = sorted(os.listdir(prefix))
        frame_path = frame_path = os.path.join(prefix, frames[-1])
        frame = cv2.imread(frame_path)         
    
    frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]), cfg.TRAIN.CROP_SIZE, random_idx)
       
    if flipped:
        frame = frame[:, ::-1, :]

    if DEBUG:
        cv2.imshow('frame', frame/255.0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return frame        

def _get_video_blob(roidb, scale_inds, phase='train'):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """
    processed_videos = []
    
    for i,item in enumerate(roidb):
        # just one scale implementated
        video_length = cfg.TRAIN.LENGTH[scale_inds[0]]  
        video = np.zeros((video_length, cfg.TRAIN.CROP_SIZE,
                        cfg.TRAIN.CROP_SIZE, 3))
        j = 0

        if phase == 'train':
            random_idx = [np.random.randint(cfg.TRAIN.FRAME_SIZE[1]-cfg.TRAIN.CROP_SIZE),
                            np.random.randint(cfg.TRAIN.FRAME_SIZE[0]-cfg.TRAIN.CROP_SIZE)]
            # TODO: data argumentation
            #image_w, image_h, crop_w, crop_h = cfg.TRAIN.FRAME_SIZE[1], cfg.TRAIN.FRAME_SIZE[0], cfg.TRAIN.CROP_SIZE, cfg.TRAIN.CROP_SIZE
            #offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h) 
            #random_idx = offsets[ npr.choice(len(offsets)) ]
        else:
            random_idx = [int((cfg.TRAIN.FRAME_SIZE[1]-cfg.TRAIN.CROP_SIZE) / 2), 
                      int((cfg.TRAIN.FRAME_SIZE[0]-cfg.TRAIN.CROP_SIZE) / 2)]
                                      
        if DEBUG:
            print ("offsets: {}, random_idx: {}".format(offsets, random_idx))
            
        video_info = item['frames'][0] #for video_info in item['frames']:
        step = video_info[3] if cfg.INPUT=='video' else 1
        prefix = item['fg_name'] if video_info[0] else item['bg_name']
        
        if cfg.TEMP_SPARSE_SAMPLING:       
            if phase == 'train':
                segment_offsets = npr.randint(step, size=len(range(video_info[1], video_info[2], step)))
            else:
                segment_offsets = np.zeros(len(range(video_info[1], video_info[2], step))) + step // 2
        else:            
            segment_offsets = np.zeros(len(range(video_info[1], video_info[2], step)))

        for i, idx in enumerate(range(video_info[1], video_info[2], step)):
            frame_idx = int(segment_offsets[i]+idx+1)            
            frame_path = os.path.join(prefix, 'image_'+str(frame_idx).zfill(5)+'.jpg')
            frame = cv2.imread(frame_path)
            # process the boundary frame
            if frame is None:          
                frames = sorted(os.listdir(prefix))
                frame_path = frame_path = os.path.join(prefix, frames[-1])
                frame = cv2.imread(frame_path)         
            
            frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]), cfg.TRAIN.CROP_SIZE, random_idx)
               
            if item['flipped']:
                frame = frame[:, ::-1, :]

            if DEBUG:
                cv2.imshow('frame', frame/255.0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            video[j] = frame
            j = j + 1
            
        video[j:video_length] = video[j-1]
        
    processed_videos.append(video)
    # Create a blob to hold the input images, dimension trans CLHW
    blob = video_list_to_blob(processed_videos)

    return blob
