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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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

def generate_classes(meta_file):
    #META_FILE = '../../preprocess/activitynet/activity_net.v1-3.min.json'
    data = json.load(open(meta_file))
    class_list = []
    for vid, vinfo in data['database'].items():
        for item in vinfo['annotations']:
            class_list.append(item['label'])

    class_list = list(set(class_list))
    class_list = sorted(class_list)
    classes = {0: 'Background'}
    for i,cls in enumerate(class_list):
        classes[i+1] = cls
    return classes

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
            ave_scores = np.zeros(201)
            for i in range(1, 201):
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
                tmp = {'label':classes[lab], 'score':nms_in[i][2], 'segment': nms_in[i][0:2]}
                res_nms.append(tmp)
          
        # select topk
        scores = [d['score'] for d in res_nms]
        sortid = np.argsort(scores)[-topk:]
        res[vid] = [res_nms[id] for id in sortid]
    return res

parser = argparse.ArgumentParser(description="log analysis.py")
parser.add_argument('log_file', type=str, help="test log file path")
parser.add_argument('--framerate', type=int, help="frame rate of videos extract by ffmpeg")
parser.add_argument('--thresh', type=float, default=0.0005, help="filter those dets low than the thresh, default=0.0005")
parser.add_argument('--nms_thresh', type=float, default=0.3, help="nms thresh, default=0.3")
parser.add_argument('--topk', type=int, default=200, help="select topk dets, default=200")
parser.add_argument('--num_cls', type=int, default=0, help="select most likely classes, default=0")

parser.add_argument('--ground_truth_filename', type=str, default='./Evaluation/data/activity_net.v1-3.min.json',
               help='Full path to json file containing the ground truth.')
parser.add_argument('--prediction_filename', type=str, default='results.json',
               help='Full path to json file containing the predictions.')
parser.add_argument('--subset', default='validation',
               help=('String indicating subset to evaluate: '
                     '(training, validation)'))
parser.add_argument('--tiou_thresholds', type=float, default=np.linspace(0.5, 0.95, 10),
               help='Temporal intersection over union threshold.')
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--check_status', type=bool, default=True)

args = parser.parse_args()
args.ground_truth_filename = os.path.join(THIS_DIR, args.ground_truth_filename)

classes = generate_classes(args.ground_truth_filename)
segmentations = analysis_log(args.log_file, thresh = args.thresh, framerate=args.framerate)
segmentations = select_top(segmentations, nms_thresh=args.nms_thresh, num_cls=args.num_cls, topk=args.topk)


res = {'version': 'VERSION 1.3', 
       'external_data': {'used': True, 'details': 'C3D pre-trained on sport-1M training set'},
       'results': {}}
for vid, vinfo in segmentations.items():
    res['results'][vid] = vinfo


with open(args.prediction_filename, 'w') as outfile:
    json.dump(res, outfile)
    
from Evaluation.eval_detection import ANETdetection

def main(ground_truth_filename, prediction_filename,
         subset='validation', tiou_thresholds=np.linspace(0.5, 0.95, 10),
         verbose=True, check_status=True):

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=check_status)
    anet_detection.evaluate()

print(args)
main(args.ground_truth_filename, args.prediction_filename, args.subset, args.tiou_thresholds, args.verbose, args.check_status)    
