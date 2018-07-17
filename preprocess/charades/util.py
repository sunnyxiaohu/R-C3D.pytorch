# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import subprocess
import shutil
import os, errno
import cv2
import scipy.io
import glob
from collections import defaultdict
import h5py
import shutil
import math
import csv

def dataset_label_parser(meta_dir, split, keep_empty=False):
  class_id = defaultdict(int)
  vid_len = defaultdict(float)
  segment = {}
  with open(os.path.join(meta_dir, 'Charades_v1_{}.csv'.format(split)), 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        actions = row['actions'].split(';')
        vid_name = row['id']
        if len(actions[0]) == 0:
          print "Video {} contains No actions".format(vid_name)
          if keep_empty:
            segment[vid_name] = []
            vid_len[vid_name] = row['length']
          continue;

        segment[vid_name] = []
        for seg in actions:
          cname = seg.split()[0]
          cid = int(cname[1:]) + 1
          start_t = float(seg.split()[1])
          end_t = float(seg.split()[2])
          class_id[cname] = cid
          segment[vid_name].append([start_t, end_t, class_id[cname]])
        vid_len[vid_name] = row['length']

  # sort segments by start_time
  for vid in segment:
    segment[vid].sort(key=lambda x: x[0])

  if True:
    keys = segment.keys()
    keys.sort()
    with open('segment.txt', 'w') as f:
      for k in keys:
        f.write("{}\n{}\n\n".format(k,segment[k]))

  return segment

def get_segment_len(segment):
  segment_len = []
  for vid_seg in segment.values():
    for seg in vid_seg:
      l = seg[1] - seg[0]
      assert l > 0
      segment_len.append(l)
  return segment_len

def mkdir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise

def rm(path):
  try:
    shutil.rmtree(path)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise

def ffmpeg(filename, outfile, fps):
  command = ["ffmpeg", "-i", filename, "-q:v", "1", "-r", str(fps), outfile]
  pipe = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  pipe.communicate()


def resize(filename, size = (171, 128)):
  img = cv2.imread(filename, 100)
  img2 = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
  cv2.imwrite(filename, img2, [100])

