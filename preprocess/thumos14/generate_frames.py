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
VIDEO_DIR = '/media/G/DataSets/THUMOS'
FRAME_DIR = '/media/F/THUMOS14'

META_DIR = os.path.join(FRAME_DIR, 'annotation_')

def generate_frame(split):
  SUB_FRAME_DIR = os.path.join(FRAME_DIR, split)
  mkdir(SUB_FRAME_DIR)
  segment = dataset_label_parser(META_DIR+split, split, use_ambiguous=True)
  video_list = segment.keys()
  for vid in video_list:
    filename = os.path.join(VIDEO_DIR, split, vid+ext)
    outpath = os.path.join(FRAME_DIR, split, vid)
    outfile = os.path.join(outpath, "image_%5d.jpg")
    mkdir(outpath)
    ffmpeg(filename, outfile, fps)
    for framename in os.listdir(outpath):
      resize(os.path.join(outpath, framename))
    frame_size = len(os.listdir(outpath))
    print (filename, fps, frame_size)

generate_frame('val')
#generate_frame('test')
#generate_frame('testing')
