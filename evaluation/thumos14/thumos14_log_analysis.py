# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import sys, os, errno
import numpy as np
import csv
import json
import copy
import argparse
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = '/media/F/THUMOS14'
META_DIR = os.path.join(FRAME_DIR, 'annotation_')

def nms(dets, thresh=0.4):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]
    lengths = x2 - x1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def generate_classes(meta_dir, split, use_ambiguous=False):
    class_id = {0: 'Background'}
    with open(os.path.join(meta_dir, 'detclasslist.txt'), 'r') as f:
        lines = f.readlines()
        for l in lines:
            cname = l.strip().split()[-1]
            cid = int(l.strip().split()[0])
            class_id[cid] = cname
        if use_ambiguous:
            class_id[21] = 'Ambiguous'

    return class_id
'''
def get_segments(data, thresh, framerate):
    segments = []
    vid = 'Background'
    find_next = False
    tmp = {'label' : 0, 'score': 0, 'segment': [0, 0]}
    for l in data:
      # video name and sliding window length
      if "fg_name :" in l:
         vid = l.split('/')[-1]

      # frame index, time, confident score
      elif "frames :" in l:
         start_frame=int(l.split()[4])
         end_frame=int(l.split()[5])
         stride = int(l.split()[6].split(']')[0])

      elif "activity:" in l:
         label = int(l.split()[1])
         tmp['label'] = label
         find_next = True

      elif "im_detect" in l:
         return vid, segments

      elif find_next:
         try: 
           left_frame = float(l.split()[0].split('[')[-1])*stride + start_frame
           right_frame = float(l.split()[1])*stride + start_frame
         except:
           left_frame = float(l.split()[1])*stride + start_frame
           right_frame = float(l.split()[2])*stride + start_frame
         if (left_frame < end_frame) and (right_frame <= end_frame):
           left  = left_frame / 25.0
           right = right_frame / 25.0
           try: 
             score = float(l.split()[-1].split(']')[0])
           except:
             score = float(l.split()[-2])
           if score > thresh:
             tmp1 = copy.deepcopy(tmp)
             tmp1['score'] = score
             tmp1['segment'] = [left, right]
             segments.append(tmp1)
         elif (left_frame < end_frame) and (right_frame > end_frame):
             if (end_frame-left_frame)*1.0/(right_frame-left_frame)>=0:
                 right_frame = end_frame
                 left  = left_frame / 25.0
                 right = right_frame / 25.0
                 try: 
                   score = float(l.split()[-1].split(']')[0])
                 except:
                   score = float(l.split()[-2])
                 if score > thresh:
                     tmp1 = copy.deepcopy(tmp)
                     tmp1['score'] = score
                     tmp1['segment'] = [left, right]
                     segments.append(tmp1)

'''
def get_segments(data, thresh, framerate):
    segments = []
    vid = 'Background'
    find_next = False
    tmp = {'label' : 0, 'score': 0, 'segment': [0, 0]}
    for l in data:
        # video name and sliding window length
        if "fg_name:" in l:
            vid = l.split('/')[-1]

        # frame index, time, confident score
        elif "frames:" in l:
            start_frame=int(l.split()[3])
            end_frame=int(l.split()[4])
            stride = int(l.split()[5].split(']')[0])

        elif "activity:" in l:
            label = int(l.split()[1])
            tmp['label'] = label
            find_next = True

        elif "im_detect" in l:
            return vid, segments

        elif find_next:
            try: 
                left_frame = float(l.split()[0].split('[')[-1])*stride + start_frame
                right_frame = float(l.split()[1])*stride + start_frame               
            except:
                left_frame = float(l.split()[1])*stride + start_frame
                right_frame = float(l.split()[2])*stride + start_frame

            try:
                score = float(l.split()[-1].split(']')[0])                
            except:
                score = float(l.split()[-2])    
                            
            if (left_frame >= right_frame):
                print("???", l)
                continue
                
            if right_frame > end_frame:
                #print("right out", right_frame, end_frame)
                right_frame = end_frame
                                
            left  = left_frame / framerate
            right = right_frame / framerate                
            if score > thresh:
                tmp1 = copy.deepcopy(tmp)
                tmp1['score'] = score
                tmp1['segment'] = [left, right]
                segments.append(tmp1)
                
def analysis_log(logfile, thresh, framerate):
    with open(logfile, 'r') as f:
        lines = f.read().splitlines()
    predict_data = []
    res = {}
    for l in lines:
        if "frames:" in l:
            predict_data = []
        predict_data.append(l)
        if "im_detect:" in l:
            vid, segments = get_segments(predict_data, thresh, framerate)
            if vid not in res:
                res[vid] = []
            res[vid] += segments
    return res

def select_top(segmentations, nms_thresh=0.99999, num_cls=0, topk=0):
  res = {}
  for vid, vinfo in segmentations.items():
    # select most likely classes
    if num_cls > 0:
      ave_scores = np.zeros(21)
      for i in xrange(1, 21):
        ave_scores[i] = np.sum([d['score'] for d in vinfo if d['label']==i])
      labels = list(ave_scores.argsort()[::-1][:num_cls])
    else:
      labels = list(set([d['label'] for d in vinfo]))

    # NMS
    res_nms = []
    for lab in labels:
      nms_in = [d['segment'] + [d['score']] for d in vinfo if d['label'] == lab]
      keep = nms(np.array(nms_in), nms_thresh)
      for i in keep:
        # tmp = {'label':classes[lab], 'score':nms_in[i][2], 'segment': nms_in[i][0:2]}
        tmp = {'label': lab, 'score':nms_in[i][2], 'segment': nms_in[i][0:2]}
        res_nms.append(tmp)
      
    # select topk
    scores = [d['score'] for d in res_nms]
    sortid = np.argsort(scores)[-topk:]
    res[vid] = [res_nms[id] for id in sortid]
  return res

parser = argparse.ArgumentParser(description="log analysis.py")
parser.add_argument('log_file', type=str, help="test log file path")
parser.add_argument('--framerate', type=int, help="frame rate of videos extract by ffmpeg")
parser.add_argument('--thresh', type=float, default=0.005, help="filter those dets low than the thresh, default=0.0005")
parser.add_argument('--nms_thresh', type=float, default=0.4, help="nms thresh, default=0.3")
parser.add_argument('--topk', type=int, default=200, help="select topk dets, default=200")
parser.add_argument('--num_cls', type=int, default=0, help="select most likely classes, default=0")  

args = parser.parse_args()
classes = generate_classes(META_DIR+'test', 'test', use_ambiguous=False)
segmentations = analysis_log(args.log_file, thresh = args.thresh, framerate=args.framerate)
segmentations = select_top(segmentations, nms_thresh=args.nms_thresh, num_cls=args.num_cls, topk=args.topk)


res = {'version': 'VERSION 1.3', 
       'external_data': {'used': True, 'details': 'C3D pre-trained on activity-1.3 training set'},
       'results': {}}
for vid, vinfo in segmentations.items():
  res['results'][vid] = vinfo

#with open('results.json', 'w') as outfile:
#  json.dump(res, outfile)

with open('tmp.txt', 'w') as outfile:
  for vid, vinfo in segmentations.items():
    for seg in vinfo:
      outfile.write("{} {} {} {} {}\n".format(vid, seg['segment'][0], seg['segment'][1], int(seg['label']) ,seg['score']))
      
      
def matlab_eval():
    print('Computing results with the official Matlab eval code')
    path = os.path.join(THIS_DIR, 'Evaluation')
    cmd = 'cp tmp.txt {} && '.format(THIS_DIR)
    cmd += 'cd {} && '.format(path)
    cmd += 'matlab -nodisplay -nodesktop '
    cmd += '-r "dbstop if error; '
    cmd += 'eval_thumos14(); quit;"'
    
    print('Runing: \n {}'.format(cmd))
    status = subprocess.call(cmd, shell=True)
    
matlab_eval()
