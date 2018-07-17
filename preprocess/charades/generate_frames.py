#coding=utf-8
# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
from util import *
import json
import glob

fps = 25
ext = '.mp4'
VIDEO_DIR = '/media/agwang/新加卷/DataSets/Charades/Charades_v1_480'
FRAME_DIR = '/media/agwang/03c94b1e-c46c-4c7b-8d3f-47e316fdee74/home/ksnzh/Videos/action-datasets/Charades'

META_DIR = os.path.join(VIDEO_DIR, '../Charades_meta')

def generate_frame(split, keep_empty):
  SUB_FRAME_DIR = os.path.join(FRAME_DIR, split)
  mkdir(SUB_FRAME_DIR)
  segment = dataset_label_parser(META_DIR, split, keep_empty)
  video_list = segment.keys()

  for vid in video_list:
    filename = os.path.join(VIDEO_DIR, vid+ext)
    outpath = os.path.join(FRAME_DIR, split, vid)
    outfile = os.path.join(outpath, "image_%5d.jpg")
    mkdir(outpath)
    ffmpeg(filename, outfile, fps)
    for framename in os.listdir(outpath):
      resize(os.path.join(outpath, framename))
    frame_size = len(os.listdir(outpath))
    print filename, fps, frame_size

generate_frame('train', keep_empty=False)
generate_frame('test', keep_empty=True)
#generate_frame('testing')
