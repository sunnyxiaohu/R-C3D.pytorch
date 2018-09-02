# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
from util import *
import json
from joblib import delayed
from joblib import Parallel
from multiprocessing import Pool
import pandas as pd

FPS = 5

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
        ffmpeg(filename, outfile, FPS)
        for framename in os.listdir(outpath):
          resize(os.path.join(outpath, framename))
        frame_size = len(os.listdir(outpath))
        print (filename, duration, FPS, frame_size)

generate_frame('training')
generate_frame('validation')
#generate_frame('testing')
'''
# For parallel
file_list = []
for split in ['training', 'validation']:
    for vid, vinfo in meta_data['database'].items():
        if vinfo['subset'] == split:
            # make sure the video has been downloaded.
            vname = [s for s in video_list if vid in s]
            if len(vname) >0:
                file_list.append(vname)
print("{} videos needed to be extracted".format(len(file_list)))    

def generate_frame_wrapper(vid, vinfo, split):
    #print(vid, vinfo)
    if vinfo['subset'] == split:
        # make sure the video has been downloaded.
        vname = [s for s in video_list if vid in s]
        if len(vname) != 0 :
            filename = VIDEO_DIR + vname[0]
            duration = vinfo['duration']
            outpath = os.path.join(FRAME_DIR, split, vid)
            outfile = os.path.join(outpath, "image_%5d.jpg")
            if os.path.exists(outpath) is False:
                mkdir(outpath)
                ffmpeg(filename, outfile, FPS)
                for framename in os.listdir(outpath):
                    resize(os.path.join(outpath, framename))
                frame_size = len(os.listdir(outpath))
                print (filename, duration, FPS, frame_size)


for split in ["training", "validation"]:
    SUB_FRAME_DIR = os.path.join(FRAME_DIR, split)
    mkdir(SUB_FRAME_DIR)
    status_lst = Parallel(n_jobs=8)(delayed(generate_frame_wrapper)(vid, vinfo, split) for vid, vinfo in meta_data["database"].items())
'''    
