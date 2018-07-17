# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
from util import *
import json

fps = 5

VIDEO_DIR = '/media/agwang/ShiGuangB/DataSets/ActivityNet/Videos/'
video_list = os.listdir(VIDEO_DIR)

META_FILE = './activity_net.v1-3.min.json'
meta_data = json.load(open(META_FILE))

FRAME_DIR = '/media/agwang/Data1/action-datasets/ActivityNet/frames/'
mkdir(FRAME_DIR)

def generate_frame(split):
  SUB_FRAME_DIR = os.path.join(FRAME_DIR, split)
  mkdir(SUB_FRAME_DIR)
  for vid, vinfo in meta_data['database'].items():
    if vinfo['subset'] == split:
      vname = [s for s in video_list if vid in s]
      if len(vname) != 0 :
        filename = VIDEO_DIR + vname[0]
        duration = vinfo['duration']
        outpath = os.path.join(SUB_FRAME_DIR, vid)
        outfile = os.path.join(outpath, "image_%5d.jpg")
        if os.path.exists(outpath):
          continue
        mkdir(outpath)
        ffmpeg(filename, outfile, fps)
        for framename in os.listdir(outpath):
          resize(os.path.join(outpath, framename))
        frame_size = len(os.listdir(outpath))
        print (filename, duration, fps, frame_size)

generate_frame('training')
generate_frame('validation')
#generate_frame('testing')
